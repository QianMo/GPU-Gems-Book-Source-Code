/**
 @file NetworkDevice.h

 These classes abstract networking from the socket level to a serialized messaging
 style that is more appropriate for games.  The performance has been tuned for 
 sending many small messages.  The message protocol contains a header that prevents
 them from being used with raw UDP/TCP (e.g. connecting to an HTTP server).  A 
 future stream API will address this.

 LightweightConduit and ReliableConduits have different interfaces because
 they have different semantics.  You would never want to interchange them without
 rewriting the surrounding code.

 NetworkDevice creates conduits because they need access to a global log pointer
 and because I don't want non-reference counted conduits being created.

 Be careful with threads and reference counting.  The reference counters are not
 threadsafe, and are also not updated correctly if a thread is explicitly killed.
 Since the conduits will be passed by const XConduitRef& most of the time this
 doesn't appear as a major problem.  With non-blocking conduits, you should need
 few threads anyway.

 LightweightConduits preceed each message with a 4-byte host order unsigned integer
 that is the message type.  This does not appear in the message
 serialization/deserialization.

 ReliableConduits preceed each message with two 4-byte host order unsigned integers.
 The first is the message type and the second indicates the length of the rest of
 the data.  The size does not include the size of the header itself.  The minimum
 message is 9 bytes, a 4-byte types, a 4-byte header of "1" and one byte of data.

 @maintainer Morgan McGuire, morgan@graphics3d.com
 @created 2002-11-22
 @edited  2004-01-03
 */

#ifndef NETWORKDEVICE_H
#define NETWORKDEVICE_H

#include "G3D/platform.h"
#include <string>
#include "G3D/g3dmath.h"
#ifdef _WIN32
    #include <winsock.h>
#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #define SOCKADDR_IN struct sockaddr_in
    #define SOCKET int
#endif
#include "G3D/ReferenceCount.h"
#include "G3D/Array.h"

namespace G3D {

class NetAddress {
private:
    friend class NetworkDevice;
    friend class LightweightConduit;
    friend class ReliableConduit;

    /** Host byte order */
    void init(uint32 host, uint16 port);
    void init(const std::string& hostname, uint16 port);
    NetAddress(const SOCKADDR_IN& a);
    NetAddress(const struct in_addr& addr, uint16 port = 0);

    SOCKADDR_IN                 addr;

public:
    /**
     In host byte order
     */
    NetAddress(uint32 host, uint16 port = 0);

    /**
     Port is in host byte order.
     */
    NetAddress(const std::string& hostname, uint16 port = 0);

    /**
    String must be in the form "hostname:port"
     */
    NetAddress(const std::string& hostnameAndPort);

    /**
     For use with a lightweight conduit.
     */
    static NetAddress broadcastAddress(uint16 port);

    NetAddress();

    void serialize(class BinaryOutput& b) const;
    void deserialize(class BinaryInput& b);

    /** Returns true if this is not an illegal address. */
    bool ok() const;

    /** Returns a value in host format */
    inline uint32 ip() const {
        return ntohl(addr.sin_addr.s_addr);
        //return ntohl(addr.sin_addr.S_un.S_addr);
    }

    inline uint16 port() const {
        return ntohs(addr.sin_port);
    }

    std::string ipString() const;
    std::string toString() const;

};


inline unsigned int hashCode(const NetAddress& a) {
	return a.ip() + ((uint32)a.port() << 16);
}


/**
 Two addresses may point to the same computer but be != because
 they have different IP's.
 */
inline bool operator==(const NetAddress& a, const NetAddress& b) {
	return (a.ip() == b.ip()) && (a.port() == b.port());
}


inline bool operator!=(const NetAddress& a, const NetAddress& b) {
    return !(a == b);
}


/**
 Interface for data sent through a conduit.
 */
class NetMessage {
public:
    virtual ~NetMessage() {}

    /** This must return a value even for an uninitalized method.
       Create an enumeration for your message types and return
       one of those values.  It will be checked on both send and
       receive as a form of runtime type checking. 
    
       Values less than 1000 are reserved for the system.*/
    virtual uint32 type() const = 0;
    virtual void serialize(class BinaryOutput& b) const = 0;
    virtual void deserialize(class BinaryInput& b) = 0;
};


class Conduit : public ReferenceCountedObject {
protected:
    friend class NetworkDevice;
    friend class NetListener;

    uint64                          mSent;
    uint64                          mReceived;
    uint64                          bSent;
    uint64                          bReceived;

    class NetworkDevice*            nd;
    SOCKET                          sock;

    Conduit(class NetworkDevice* _nd);

public:

    virtual ~Conduit();
    uint64 bytesSent() const;
    uint64 messagesSent() const;
    uint64 bytesReceived() const;
    uint64 messagesReceived() const;

    /**
     If true, receive will return true.
     */
    virtual bool messageWaiting() const;

    /**
     Returns the type of the waiting message (i.e. the type supplied with send).
     The return value is zero when there is no message waiting.

     One way to use this is to have a Table mapping message types to
     pre-allocated NetMessage subclasses so receiving looks like:

     <PRE>
         // My base class for messages.
         class Message : public NetMessage {
             virtual void process() = 0;
         };

         Message* m = table[conduit->waitingMessageType()];
         conduit->receive(m);
         m->process();
     </PRE>

      Another is to simply SWITCH on the message type.
     */
    virtual uint32 waitingMessageType() = 0;

    /** Returns true if the connection is ok. */
    bool ok() const;
};


// Messaging and stream APIs must be supported on a single class because
// sometimes an application will switch modes on a single socket.  For
// example, when transferring 3D level geometry during handshaking with
// a game server.
class ReliableConduit : public Conduit {
private:
    friend class NetworkDevice;
    friend class NetListener;

    NetAddress                      addr;
    /**
     True when the messageType has been read but the
     packet has not been read.
     */
    bool                            alreadyReadType;
    
    /**
     Type of the incoming message.
     */
    uint32                          messageType;

    ReliableConduit(class NetworkDevice* _nd, const NetAddress& addr);

    ReliableConduit(class NetworkDevice* _nd, const SOCKET& sock, const NetAddress& addr);

public:

    /** Closes the socket. */
    ~ReliableConduit();

    /**
     Serializes the message and schedules it to be sent as soon as possible,
     then returns immediately.

     The actual data sent across the network is preceeded by the message type
     and the size of the serialized message as a 32-bit integer.  The size is
     sent because TCP is a stream protocol and doesn't have a concept of discrete
     messages.
     */
    void send(const NetMessage* m);

    virtual uint32 waitingMessageType();

    /** If a message is waiting, deserializes the waiting message into m and
        returns true, otherwise returns false.  
        
        If a message is incoming but was split across multipled TCP packets
        in transit, this will block for up to .25 seconds waiting for all
        packets to arrive.  For short messages (less than 5k) this is extremely
        unlikely to occur.*/
    bool receive(NetMessage* m);

    NetAddress address() const;
};


typedef ReferenceCountedPointer<class ReliableConduit> ReliableConduitRef;


/**
 Provides fast but unreliable transfer of messages.  LightweightConduits
 are implemented using UDP.  On a LAN messages are extremely likely to arrive.
 On the internet, some messages may be dropped.  The receive order of successively sent
 messages is not guaranteed.
 */
class LightweightConduit : public Conduit {
private:
    friend class NetworkDevice;

    /**
     True when waitingForMessageType has read the message
     from the network into messageType/messageStream.
     */
    bool                    alreadyReadMessage;

    /**
     Origin of the received message.
     */
    NetAddress              messageSender;

    /**
     The type of the last message received.
     */
    uint32                  messageType;

    /**
     The message received (the type has already been read off).
     */
    Array<uint8>            messageBuffer;

    LightweightConduit(class NetworkDevice* _nd, uint16 receivePort, bool enableReceive, bool enableBroadcast);

public:

    /** Closes the socket. */
    ~LightweightConduit();

    /** Serializes and sends the message immediately. Data may not arrive and may
        arrive out of order, but individual messages are guaranteed to not be
        corrupted.  If the message is null, an empty message is still sent.*/
    void send(const NetAddress& a, const NetMessage* m);

    /** If data is waiting, deserializes the waiting message into m, puts the
        sender's address in addr and returns true, otherwise returns false.  
        If m is NULL, the message is consumed but not deserialized.
    */
    bool receive(NetMessage* m, NetAddress& sender);

    virtual uint32 waitingMessageType();

    virtual bool messageWaiting() const;
};

typedef ReferenceCountedPointer<class LightweightConduit> LightweightConduitRef;

//////////////////////////////////////////////////////////////////////////////////

/**
 Runs on the server listening for clients trying to make reliable connections.
 */
class NetListener : public ReferenceCountedObject {
private:

    friend class NetworkDevice;

    class NetworkDevice*            nd;
    SOCKET                          sock;

    /** Port is in host byte order. */
    NetListener(class NetworkDevice* _nd, uint16 port);

public:

    ~NetListener();

    /** Block until a connection is received.  Returns NULL if 
        something went wrong. */
    ReliableConduitRef waitForConnection();

    /** True if a client is waiting (i.e. waitForConnection will return immediately). */
    bool clientWaiting() const;

    bool ok() const;
};

typedef ReferenceCountedPointer<class NetListener> NetListenerRef;

//////////////////////////////////////////////////////////////////////////////////

/**
 An abstraction over sockets that provides a message based network infrastructure
 optimized for sending many small (>500 byte) messages.  All functions always return
 immediately.
 */
class NetworkDevice {
private:
    friend class Conduit;
    friend class LightweightConduit;
    friend class ReliableConduit;
    friend class NetListener;

    class Log*                  debugLog;

    bool                        initialized;

    /** Utility method. */
    void closesocket(SOCKET& sock) const;

    /** Utility method. */
    void bind(SOCKET sock, const NetAddress& addr) const;

public:

    NetworkDevice();

    /**
     Returns the log this was initialized with.
     */
    Log* log() const {
        return debugLog;
    }

    /** Returns the name of this computer */
    std::string localHostName() const;

    /** There is often more than one address for the local host. This returns all of them. */
    void localHostAddresses(Array<NetAddress>& array) const;

    /**
     Returns false if there was a problem initializing the network.
     */
    bool init(class Log* log = NULL);

    /**
     Shuts down the network device.
     */
    void cleanup();

    /**
     If receivePort is specified and enableReceive is true, the conduit can 
     receive as well as send.
     @param receivePort host byte order
     */
    LightweightConduitRef createLightweightConduit(uint16 receivePort = 0, bool enableReceive = false, bool enableBroadcast = false);

    /**
     Client invokes this to connect to a server.  The call blocks until the 
     conduit is opened.  The conduit will not be ok() if it fails.
     */
    ReliableConduitRef createReliableConduit(const NetAddress& address);

    /**
     Call this on the server side to create an object that listens for
     connections.
     */
    NetListenerRef createListener(const uint16 port);
};

}

#ifndef _WIN32
#undef SOCKADDR_IN
#undef SOCKET
#endif

#endif
