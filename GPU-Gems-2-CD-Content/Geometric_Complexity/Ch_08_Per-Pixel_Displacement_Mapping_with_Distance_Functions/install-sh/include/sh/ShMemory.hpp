// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Michael D. McCool
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must
// not claim that you wrote the original software. If you use this
// software in a product, an acknowledgment in the product documentation
// would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must
// not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
// distribution.
//////////////////////////////////////////////////////////////////////////////
#ifndef SHMEMORY_HPP
#define SHMEMORY_HPP

#include <list>
#include <map>
#include <utility>
#include <string>
#include "ShDllExport.hpp"
#include "ShRefCount.hpp"
#include "ShMemoryDep.hpp"

namespace SH {

/** @defgroup memory Buffer Management
 * @{
 */

class ShStorage;

/** A memory object.
 * A memory object represents a chunk of data.   Note, however, that
 * copies of this data may be stored in more than one place, i.e. on
 * both the CPU and the GPU.   This class keeps track of the "current"
 * version and makes transfers as necessary.
 * @todo Maybe keep a length field (in bytes)?
 * @see ShStorage
 */
class
SH_DLLEXPORT ShMemory : public ShRefCountable {
public:
  virtual ~ShMemory();

  /// Return the timestamp of the most up-to-date storage of this memory
  int timestamp() const;

  /// Find a storage of a given id.
  ShPointer<ShStorage> findStorage(const std::string& id);

  /// Find a storage of a given id selected by an external functor.
  /// @arg functor A functor of ShStoragePtr -> bool.
  template<typename Functor>
  ShPointer<ShStorage> findStorage(const std::string& id, const Functor& f);

  /// Discard this storage from this memory
  void removeStorage(const ShPointer<ShStorage>& storage);

  /// link the memory to a specific dependency
  void add_dep(ShMemoryDep* dep);

  /// modify the memory data by using the links to the dependencies
  void flush();

protected:
  ShMemory();
  
private:
  void updateTimestamp(int timestamp);

  void addStorage(const ShPointer<ShStorage>& storage);

  typedef std::list< ShPointer<ShStorage> > StorageList;
  StorageList m_storages;
  int m_timestamp;

  /// the list of all dependencies, for update calls
  std::list<ShMemoryDep*> dependencies;

  friend class ShStorage;

  /// NOT IMPLEMENTED
  ShMemory& operator=(const ShMemory& other);
  /// NOT IMPLEMENTED
  ShMemory(const ShMemory& other);
};

typedef ShPointer<ShMemory> ShMemoryPtr;
typedef ShPointer<const ShMemory> ShMemoryCPtr;

/** A Storage Transfer function.
 * A transfer specifies the cost and manner of transferring data from
 * one storage to another.
 * Static instances of these should be created for any possible
 * storage transfers.
 * Use the ShStorage::addTransfer() function to add a transfer to the
 * list.
 *
 * @see ShMemory
 * @see ShStorage
 */
class
SH_DLLEXPORT ShTransfer {
public:
  /// Returns true if the transfer succeeded.
  virtual bool transfer(const ShStorage* from, ShStorage* to) = 0;

  /// Returns -1 if transfer impossible, otherwise the cost of
  /// transfer.
  /// @todo Perhaps this needs more information, e.g. size of the storages.
  virtual int cost() = 0;

protected:
  ShTransfer(const std::string& from, const std::string& to);

private:
  
  /// NOT IMPLEMENTED
  ShTransfer(const ShTransfer& other);
  /// NOT IMPLEMENTED
  ShTransfer& operator=(const ShTransfer& other);
};

/** A Storage object
 * Storage represents actual, physical storage of some memory's
 * contents. This might be on the CPU, the GPU, a filesystem or even
 * across the network.
 *
 * @see ShMemory
 * @see ShTransfer
 */
class
SH_DLLEXPORT ShStorage : public ShRefCountable {
public:
  ShStorage();
  virtual ~ShStorage();

  /// Return the version of the data currently stored by this storage
  int timestamp() const;
  /// Force-set the version of the data currently stored by this storage
  void setTimestamp(int timestamp);
  /// Return the memory this storage represents
  const ShMemory* memory() const;
  /// Return the memory this storage represents
  ShMemory* memory();

  /// Make sure this storage is in sync with the latest version of the
  /// memory.
  /// @todo Maybe make this const... after all, we need to do this on a
  /// read!
  void sync();

  /// Mark an upcoming write to this storage.
  /// This will sync, if necessary.
  void dirty();
  
  // Mark an upcoming write to this storage.
  /// Don't call sync, all storages are replaced.
  void dirtyall();
  
  /// Return an id uniquely identifying the _type_ of this storage
  /// This is used for looking up transfer functions.
  /// e.g.: host, opengl:texture, sm:texture
  virtual std::string id() const = 0;
  
  /// Return the cost of transferring from one storage to another.
  /// Returns -1 if there is no possible transfer.
  static int cost(ShStorage* from, ShStorage* to);

  /// Transfer data from one storage to another.
  /// Returns true if the transfer succeeded.
  static bool transfer(ShStorage* from, ShStorage* to);

  /// Use this to register new transfer functions when they are instantiated.
  static void addTransfer(const std::string& from,
                          const std::string& to,
                          ShTransfer* transfer);

  /// Called by ShMemory when it destructs. Necessary for refcounting purposes.
  void orphan();
  
protected:
  ShStorage(ShMemory* memory);

private:
  ShMemory* m_memory;
  int m_timestamp;

  typedef std::map<std::pair<std::string, std::string>, ShTransfer*> TransferMap;
  static TransferMap* m_transfers;

  /// NOT IMPLEMENTED
  ShStorage(const ShStorage& other);
  /// NOT IMPLEMENTED
  ShStorage& operator=(const ShStorage& other);
};

typedef ShPointer<ShStorage> ShStoragePtr;
typedef ShPointer<const ShStorage> ShStorageCPtr;

/** An ShStorage representing data stored on the CPU host memory.
 * You probably want to use ShHostMemory to construct this.
 * @see ShMemory
 * @see ShStorage
 * @see ShHostMemory
 */
class
SH_DLLEXPORT ShHostStorage : public ShStorage {
public:
  ShHostStorage(ShMemory* memory, int length); ///< Internally managed storage
  ShHostStorage(ShMemory* memory, int length, void* data); ///< Externally managed storage

  /// Destruct this storage.
  /// If the memory was allocated internally, it will be freed automatically.
  ~ShHostStorage();

  std::string id() const;

  /// Return the length (in bytes) of data represented by this storage.
  int length() const;

  /// Return the location of the storage's data on the host
  const void* data() const;
  /// Return the location of the storage's data on the host
  void* data();
  
private:
  int m_length; ///< number of bytes stored
  void* m_data; ///< the actual data, stored on the host

  bool m_managed; ///< Did we create the data? If so, this is true

  // NOT IMPLEMENTED
  ShHostStorage& operator=(const ShHostStorage& other);
  ShHostStorage(const ShHostStorage& other);
};

typedef ShPointer<ShHostStorage> ShHostStoragePtr;
typedef ShPointer<const ShHostStorage> ShHostStorageCPtr;

/** An ShMemory initially originating in CPU host memory.
 * @see ShMemory
 * @see ShHostStorage
 */
class
SH_DLLEXPORT ShHostMemory : public ShMemory {
public:
  ShHostMemory(int length);
  ShHostMemory(int length, void* data);

  ~ShHostMemory();

  ShHostStoragePtr hostStorage();
  ShPointer<const ShHostStorage> hostStorage() const;

private:
  ShHostStoragePtr m_hostStorage;
  // NOT IMPLEMENTED
  ShHostMemory& operator=(const ShHostMemory& other);
  ShHostMemory(const ShHostMemory& other);
};

typedef ShPointer<ShHostMemory> ShHostMemoryPtr;
typedef ShPointer<const ShHostMemory> ShHostMemoryCPtr;

template<typename Functor>
ShPointer<ShStorage> ShMemory::findStorage(const std::string& id, const Functor& f)
{
  for (StorageList::iterator I = m_storages.begin(); I != m_storages.end(); ++I) {
    if ((*I)->id() == id && f(*I)) return *I;
  }
  return 0;
}

/*@}*/

}

#endif
