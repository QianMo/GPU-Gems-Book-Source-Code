/*
Copyright (c) 2000, Sean O'Neil (s_p_oneil@hotmail.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the project nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __DateTime_h__
#define __DateTime_h__


class CDateTime;	// Forward declaration


class CTime
{
protected:
	friend class CDateTime;
	short m_nHour;
	unsigned char m_nMinute;
	unsigned char m_nSecond;

public:
	CTime()
	{
		m_nHour = 0;
		m_nMinute = 0;
		m_nSecond = 0;
	}
	CTime(int nSeconds)
	{
		SetSecondsSinceMidnight(nSeconds);
	}
	CTime(int nHour, int nMinute, int nSecond)
	{
		m_nHour = nHour;
		m_nMinute = nMinute;
		m_nSecond = nSecond;
	}
	CTime(const CTime& time)
	{
		m_nHour = time.m_nHour;
		m_nMinute = time.m_nMinute;
		m_nSecond = time.m_nSecond;
	}
	CTime(const SYSTEMTIME &st)
	{
		m_nHour = (short)st.wHour;
		m_nMinute = (unsigned char)st.wMinute;
		m_nSecond = (unsigned char)st.wSecond;
	}
	CTime(const char *psz)
	{
		ParseString(psz);
	}

	int GetHour() const					{ return m_nHour; }
	int GetMinute() const				{ return m_nMinute; }
	int GetSecond() const				{ return m_nSecond; }
	
	SYSTEMTIME GetAsSystemTime(SYSTEMTIME &st)
	{
		st.wHour = (unsigned short)m_nHour;
		st.wMinute = (unsigned short)m_nMinute;
		st.wSecond = (unsigned short)m_nSecond;
		st.wMilliseconds = 0;
		return st;
	}

	void operator=(const SYSTEMTIME &st)
	{
		m_nHour = (short)st.wHour;
		m_nMinute = (unsigned char)st.wMinute;
		m_nSecond = (unsigned char)st.wSecond;
	}
	void operator=(const CTime &time)
	{
		m_nHour = time.m_nHour;
		m_nMinute = time.m_nMinute;
		m_nSecond = time.m_nSecond;
	}
	void operator=(int nSeconds)
	{
		SetSecondsSinceMidnight(nSeconds);
	}

	int operator+(const CTime &time) const		{ return GetSecondsSinceMidnight() + time.GetSecondsSinceMidnight(); }
	int operator-(const CTime &time) const		{ return GetSecondsSinceMidnight() - time.GetSecondsSinceMidnight(); }
	void operator+=(const CTime &time)			{ *this = *this + time; }
	void operator-=(const CTime &time)			{ *this = *this - time; }

	CTime operator+(int nSeconds) const			{ return CTime(GetSecondsSinceMidnight() + nSeconds); }
	CTime operator-(int nSeconds) const			{ return CTime(GetSecondsSinceMidnight() - nSeconds); }
	void operator+=(int nSeconds)				{ *this = *this + nSeconds; }
	void operator-=(int nSeconds)				{ *this = *this - nSeconds; }

	bool operator>(const CTime &time) const			{ return Compare(time) > 0; }
	bool operator>=(const CTime &time) const		{ return Compare(time) >= 0; }
	bool operator<(const CTime &time) const			{ return Compare(time) < 0; }
	bool operator<=(const CTime &time) const		{ return Compare(time) <= 0; }
	bool operator==(const CTime &time) const		{ return (m_nHour == time.m_nHour && m_nMinute == time.m_nMinute && m_nSecond == time.m_nSecond); }
	bool operator!=(const CTime &time) const		{ return (m_nHour != time.m_nHour || m_nMinute != time.m_nMinute || m_nSecond != time.m_nSecond); }

	int Compare(const CTime &time) const
	{
		if(m_nHour > time.m_nHour)
			return 1;
		if(m_nHour < time.m_nHour)
			return -1;
		if(m_nMinute > time.m_nMinute)
			return 1;
		if(m_nMinute < time.m_nMinute)
			return -1;
		if(m_nSecond > time.m_nSecond)
			return 1;
		if(m_nSecond < time.m_nSecond)
			return -1;
		return 0;
	}

	int GetSecondsSinceMidnight() const
	{
		return m_nHour * 3600 + m_nMinute * 60 + m_nSecond;
	}

	void SetSecondsSinceMidnight(int nSeconds)
	{
		m_nHour = nSeconds / 3600;
		if(nSeconds < 0)
			m_nHour--;
		nSeconds -= m_nHour * 3600;
		m_nMinute = nSeconds / 60;
		m_nSecond = nSeconds - m_nMinute*60;
	}

	std::string toString(bool sep=true) const { 
		char buff[10];
		return GetString(buff, sep);
	}

	char *GetString(char *pszBuffer, bool bSeparator=true) const
	{
		// Build a string in HH:MM:DD format
		int nIndex = 0;
		pszBuffer[nIndex++] = m_nHour/10 + '0';
		pszBuffer[nIndex++] = m_nHour%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = ':';
		pszBuffer[nIndex++] = m_nMinute/10 + '0';
		pszBuffer[nIndex++] = m_nMinute%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = ':';
		pszBuffer[nIndex++] = m_nSecond/10 + '0';
		pszBuffer[nIndex++] = m_nSecond%10 + '0';
		pszBuffer[nIndex++] = 0;
		return pszBuffer;
	}

	bool ParseString(const char *psz)
	{
		// Parses a string in HH:MM:SS format (with or without separators)
		int nIndex = 0;
		m_nHour = (psz[nIndex++]-'0') * 10;
		m_nHour += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_nMinute = (psz[nIndex++]-'0') * 10;
		m_nMinute += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_nSecond = (psz[nIndex++]-'0') * 10;
		m_nSecond += (psz[nIndex++]-'0');
		return m_nHour >= 0 && m_nMinute >= 0 && m_nMinute <= 59 && m_nSecond >= 0 && m_nSecond <= 59;
	}
};


class CDate
{
protected:
	friend class CDateTime;
	short m_nYear;
	unsigned char m_nMonth;
	unsigned char m_nDay;

public:
	enum { Sunday = 1, Monday = 2, Tuesday = 3, Wednesday = 4, Thursday = 5, Friday = 6, Saturday = 7 };

	CDate()
	{
		m_nYear = 0;
		m_nMonth = 0;
		m_nDay = 0;
	}
	CDate(int nDays)
	{
		SetDaysSinceMillenium(nDays);
	}
	CDate(int nYear, int nMonth, int nDay)
	{
		m_nYear = nYear;
		m_nMonth = nMonth;
		m_nDay = nDay;
	}
	CDate(const CDate& date)
	{
		m_nYear = date.m_nYear;
		m_nMonth = date.m_nMonth;
		m_nDay = date.m_nDay;
	}
	CDate(const SYSTEMTIME &st)
	{
		m_nYear = (short)st.wYear;
		m_nMonth = (unsigned char)st.wMonth;
		m_nDay = (unsigned char)st.wDay;
	}
	CDate(const char *psz)
	{
		ParseString(psz);
	}

	int GetYear() const				{ return m_nYear; }
	int GetMonth() const			{ return m_nMonth; }
	int GetDay() const				{ return m_nDay; }
	int GetDayOfWeek() const		// Returns 1..7 for Sunday..Saturday
	{
		// Jan 2, 2000 was a Sunday, so calculate number of days since that day
		int nDayOffset = GetDaysSinceMillenium() - 1;	// Jan 2, 2000 is one day off from millenium
		if(nDayOffset < 0)								// Don't allow negative offsets because the mod operator may give a negative result
			nDayOffset += -nDayOffset * 7;				// Adding any multiple of 7 is safe, so this should suffice
		return (nDayOffset % 7) + 1;					// Return the mod (+1 to make the day of week 1-based)
	}
	bool IsLeapYear() const			{ return IsLeapYear(m_nYear); }
	static bool IsLeapYear(int nYear)
	{
		if(!(nYear & 0x03))
		{
			if((nYear % 400) || !(nYear % 1000))
				return true;
		}
		return false;
	}
	
	SYSTEMTIME GetAsSystemTime(SYSTEMTIME &st)
	{
		st.wYear = (unsigned short)m_nYear;
		st.wMonth = (unsigned short)m_nMonth;
		st.wDay = (unsigned short)m_nDay;
		return st;
	}

	void AddYears(int nYears)
	{
		m_nYear += nYears;
	}

	void AddMonths(int nMonths)
	{
		nMonths += m_nMonth;
		int nYears = 0;
		if(nMonths > 12)
			nYears = (nMonths - 1) / 12;
		else if(nMonths < 1)
			nYears = (nMonths - 12) / 12;
		AddYears(nYears);
		m_nMonth = (unsigned char)(nMonths - nYears * 12);
	}

	void AddDays(int nDays)
	{
		SetDaysSinceMillenium(GetDaysSinceMillenium() + nDays);
	}

	void operator=(const SYSTEMTIME &st)
	{
		m_nYear = (short)st.wYear;
		m_nMonth = (unsigned char)st.wMonth;
		m_nDay = (unsigned char)st.wDay;
	}
	void operator=(const CDate &date)
	{
		m_nYear = date.m_nYear;
		m_nMonth = date.m_nMonth;
		m_nDay = date.m_nDay;
	}
	void operator=(int nDays)
	{
		SetDaysSinceMillenium(nDays);
	}

	int operator-(const CDate &date) const			{ return GetDaysSinceMillenium() - date.GetDaysSinceMillenium(); }

	int operator+(int nDays) const					{ return GetDaysSinceMillenium() + nDays; }
	int operator-(int nDays) const					{ return GetDaysSinceMillenium() - nDays; }
	void operator+=(int nDays)						{ SetDaysSinceMillenium(GetDaysSinceMillenium() + nDays); }
	void operator-=(int nDays)						{ SetDaysSinceMillenium(GetDaysSinceMillenium() - nDays); }

	bool operator>(const CDate &date) const			{ return Compare(date) > 0; }
	bool operator>=(const CDate &date) const		{ return Compare(date) >= 0; }
	bool operator<(const CDate &date) const			{ return Compare(date) < 0; }
	bool operator<=(const CDate &date) const		{ return Compare(date) <= 0; }
	bool operator==(const CDate &date) const		{ return (m_nYear == date.m_nYear && m_nMonth == date.m_nMonth && m_nDay == date.m_nDay); }
	bool operator!=(const CDate &date) const		{ return (m_nYear != date.m_nYear || m_nMonth != date.m_nMonth || m_nDay != date.m_nDay); }

	int Compare(const CDate &date) const
	{
		if(m_nYear > date.m_nYear)
			return 1;
		if(m_nYear < date.m_nYear)
			return -1;
		if(m_nMonth > date.m_nMonth)
			return 1;
		if(m_nMonth < date.m_nMonth)
			return -1;
		if(m_nDay > date.m_nDay)
			return 1;
		if(m_nDay < date.m_nDay)
			return -1;
		return 0;
	}

	int GetDaysSinceNewYear() const
	{
		int nDaysInMonth[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
		nDaysInMonth[2] = IsLeapYear(m_nYear) ? 29 : 28;
		int nMonth = m_nMonth;
		int nCount = m_nDay - 1;
		while(nMonth > 1)
			nCount += nDaysInMonth[--nMonth];
		return nCount;
	}

	void SetDaysSinceNewYear(int nDays)
	{
		int nDaysInMonth[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
		nDaysInMonth[2] = IsLeapYear(m_nYear) ? 29 : 28;
		m_nMonth = 1;
		nDays++;
		while(nDays > nDaysInMonth[m_nMonth])
			nDays -= nDaysInMonth[m_nMonth++];
		m_nDay = (unsigned char)nDays;
	}

	int GetDaysSinceMillenium() const
	{
		int nDays = GetDaysSinceNewYear();
		int nYears = m_nYear - 2000;
		if(nYears < 0)
			nDays += 365*nYears + nYears/4;			// Simplified leap-year calculation (breaks after 400 years in either direction from 2000)
		else if(nYears > 0)
			nDays += 365*nYears + (nYears-1)/4 + 1;	// Simplified leap-year calculation (breaks after 400 years in either direction from 2000)
		return nDays;
	}

	int GetSecsSinceMillenium() const 
	{
		return GetDaysSinceMillenium() * 24 * 3600;
	}

	void SetDaysSinceMillenium(int nDays)
	{
		// Start at 2000 and try to jump to the correct year
		int nYears = (nDays / 1461) * 4;		// Simplified leap-year calculation (breaks after 400 years in either direction from 2000)
		if(nDays < 0)
			nYears -= 4;
		nDays -= (nYears/4) * 1461;
		int nTemp = (nDays-1) / 365;
		if(nTemp)
		{
			nYears += nTemp;
			nDays -= nTemp*365 + 1;
		}

		// Now that we have the correct year, determine the correct month and day
		m_nYear = 2000 + nYears;
		SetDaysSinceNewYear(nDays);
	}

	std::string toString(bool sep=true) const { 
		char buff[12];
		return GetString(buff, sep);
	}

	char *GetString(char *pszBuffer, bool bSeparator=true) const
	{
		// Build a string in YYYY-MM-DD format
		int nIndex = 0;
		pszBuffer[nIndex++] = m_nYear/1000 + '0';
		pszBuffer[nIndex++] = (m_nYear%1000)/100 + '0';
		pszBuffer[nIndex++] = (m_nYear%100)/10 + '0';
		pszBuffer[nIndex++] = m_nYear%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = '-';
		pszBuffer[nIndex++] = m_nMonth/10 + '0';
		pszBuffer[nIndex++] = m_nMonth%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = '-';
		pszBuffer[nIndex++] = m_nDay/10 + '0';
		pszBuffer[nIndex++] = m_nDay%10 + '0';
		pszBuffer[nIndex++] = 0;
		return pszBuffer;
	}

	bool ParseString(const char *psz)
	{
		// Parses a string in YYYY-MM-DD format (with or without separators)
		int nIndex = 0;
		m_nYear = (psz[nIndex++]-'0') * 1000;
		m_nYear += (psz[nIndex++]-'0') * 100;
		m_nYear += (psz[nIndex++]-'0') * 10;
		m_nYear += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_nMonth = (psz[nIndex++]-'0') * 10;
		m_nMonth += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_nDay = (psz[nIndex++]-'0') * 10;
		m_nDay += (psz[nIndex++]-'0');
		return m_nYear >= 1900 && m_nYear <= 2100 && m_nMonth >= 1 && m_nMonth <= 12 && m_nDay >= 1 && m_nDay <= 31;
	}

	bool ParseString(const char *psz, int nYearPos, int nMonthPos, int nDayPos)
	{
		// Parses a string in any format with a 4-digit year, a 2-digit month, and a 2-digit day
		m_nYear = (psz[nYearPos]-'0') * 1000 + (psz[nYearPos+1]-'0') * 100 + (psz[nYearPos+2]-'0') * 10 + (psz[nYearPos+3]-'0');
		m_nMonth = (psz[nMonthPos]-'0') * 10 + (psz[nMonthPos+1]-'0');
		m_nDay = (psz[nDayPos]-'0') * 10 + (psz[nDayPos+1]-'0');
		return m_nYear >= 1900 && m_nYear <= 2100 && m_nMonth >= 1 && m_nMonth <= 12 && m_nDay >= 1 && m_nDay <= 31;
	}
};


class CDateTime
{
protected:
	CDate m_date;
	CTime m_time;

public:
	CDateTime()
	{
	}
	CDateTime(int nYear, int nMonth, int nDay, int nHour, int nMinute, int nSecond) : m_date(nYear, nMonth, nDay), m_time(nHour, nMinute, nSecond)
	{
	}
	CDateTime(const CDate &date, const CTime &time) : m_date(date), m_time(time)
	{
		AdjustTime();
	}
	CDateTime(const CDateTime &dt) : m_date(dt.m_date), m_time(dt.m_time)
	{
	}
	CDateTime(const SYSTEMTIME &st) : m_date(st), m_time(st)
	{
	}
	CDateTime(const char *psz)
	{
		ParseString(psz);
	}

	CDateTime(int secs_since_millenium)
	{
		int days = secs_since_millenium / (24 * 3600);
		int secs = secs_since_millenium % (24 * 3600);
		m_date.SetDaysSinceMillenium(days);
		m_time.SetSecondsSinceMidnight(secs);
	}

	CDate &GetDate()				{ return m_date; }
	CTime &GetTime()				{ return m_time; }
	const CDate &GetDate() const	{ return m_date; }
	const CTime &GetTime() const	{ return m_time; }
	int GetYear() const				{ return m_date.m_nYear; }
	int GetMonth() const			{ return m_date.m_nMonth; }
	int GetDay() const				{ return m_date.m_nDay; }
	int GetHour() const				{ return m_time.m_nHour; }
	int GetMinute() const			{ return m_time.m_nMinute; }
	int GetSecond() const			{ return m_time.m_nSecond; }
	int GetDayOfWeek() const		{ return m_date.GetDayOfWeek(); }
	bool IsLeapYear() const			{ return m_date.IsLeapYear(); }

	SYSTEMTIME GetAsSystemTime(SYSTEMTIME &st)
	{
		st.wYear = (unsigned short)m_date.m_nYear;
		st.wMonth = (unsigned short)m_date.m_nMonth;
		st.wDay = (unsigned short)m_date.m_nDay;
		st.wHour = (unsigned short)m_time.m_nHour;
		st.wMinute = (unsigned short)m_time.m_nMinute;
		st.wSecond = (unsigned short)m_time.m_nSecond;
		st.wMilliseconds = 0;
		return st;
	}
	int GetSecsSinceMillenium() const 
	{
		return GetDate().GetSecsSinceMillenium() + GetTime().GetSecondsSinceMidnight();
	}

	void operator=(const SYSTEMTIME &st)
	{
		m_date = st;
		m_time = st;
	}

	void operator=(const CDateTime &dt)
	{
		m_date = dt.m_date;
		m_time = dt.m_time;
	}

	void operator=(const CDate &date)
	{
		m_date = date;
	}

	void operator=(const CTime &time)
	{
		m_time = time;
	}

	void SetCurrent()
	{
		SYSTEMTIME st;
		::GetLocalTime(&st);
		*this = st;
	}

	static CDateTime GetCurrent()
	{
		SYSTEMTIME st;
		::GetLocalTime(&st);
		return CDateTime(st);
	}

	int operator-(const CDateTime &dt) const		{ return GetSecsSinceMillenium() - dt.GetSecsSinceMillenium(); }

	CDateTime operator+(const CTime &time) const	{ return *this + time.GetSecondsSinceMidnight(); }
	CDateTime operator-(const CTime &time) const	{ return *this - time.GetSecondsSinceMidnight(); }
	void operator+=(const CTime &time)				{ *this = *this + time; }
	void operator-=(const CTime &time)				{ *this = *this - time; }

	CDateTime operator+(int nSeconds) const			{ return CDateTime(GetSecsSinceMillenium() + nSeconds); }
	CDateTime operator-(int nSeconds) const			{ return CDateTime(GetSecsSinceMillenium() - nSeconds); }
	void operator+=(int nSeconds)					{ *this = *this + nSeconds; }
	void operator-=(int nSeconds)					{ *this = *this - nSeconds; }

	bool operator>(const CDateTime &dt) const		{ return Compare(dt) > 0; }
	bool operator>=(const CDateTime &dt) const		{ return Compare(dt) >= 0; }
	bool operator<(const CDateTime &dt) const		{ return Compare(dt) < 0; }
	bool operator<=(const CDateTime &dt) const		{ return Compare(dt) <= 0; }
	bool operator==(const CDateTime &dt) const		{ return (m_date == dt.m_date && m_time == dt.m_time); }
	bool operator!=(const CDateTime &dt) const		{ return (m_date != dt.m_date || m_time != dt.m_time); }

	// Use when the CTime member has been changed dynamically and may not be between 00:00:00 and 23:59:59
	void AdjustTime()
	{
		int nDays = m_time.m_nHour / 24;
		if(m_time.m_nHour < 0)
			nDays--;
		m_date += nDays;
		m_time.m_nHour -= nDays * 24;
	}

	int Compare(const CDateTime &dt) const
	{
		int nRet = m_date.Compare(dt.m_date);
		if(nRet == 0)
			nRet = m_time.Compare(dt.m_time);
		return nRet;
	}

	std::string toString(bool sep=true) const {
		char buff[30];
		return GetString(buff, sep);
	}

	char *GetString(char *pszBuffer, bool bSeparator=true) const
	{
		// Build a string in YYYY-MM-DD HH:MM:SS format
		int nIndex = 0;
		pszBuffer[nIndex++] = m_date.m_nYear/1000 + '0';
		pszBuffer[nIndex++] = (m_date.m_nYear%1000)/100 + '0';
		pszBuffer[nIndex++] = (m_date.m_nYear%100)/10 + '0';
		pszBuffer[nIndex++] = m_date.m_nYear%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = '-';
		pszBuffer[nIndex++] = m_date.m_nMonth/10 + '0';
		pszBuffer[nIndex++] = m_date.m_nMonth%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = '-';
		pszBuffer[nIndex++] = m_date.m_nDay/10 + '0';
		pszBuffer[nIndex++] = m_date.m_nDay%10 + '0';

		if(bSeparator)
			pszBuffer[nIndex++] = ' ';
		
		pszBuffer[nIndex++] = m_time.m_nHour/10 + '0';
		pszBuffer[nIndex++] = m_time.m_nHour%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = ':';
		pszBuffer[nIndex++] = m_time.m_nMinute/10 + '0';
		pszBuffer[nIndex++] = m_time.m_nMinute%10 + '0';
		if(bSeparator)
			pszBuffer[nIndex++] = ':';
		pszBuffer[nIndex++] = m_time.m_nSecond/10 + '0';
		pszBuffer[nIndex++] = m_time.m_nSecond%10 + '0';
		pszBuffer[nIndex++] = 0;
		return pszBuffer;
	}

	bool ParseString(const char *psz)
	{
		// Parses a string in YYYY-MM-DD HH:MM:SS format (with or without separators)
		int nIndex = 0;
		m_date.m_nYear = (psz[nIndex++]-'0') * 1000;
		m_date.m_nYear += (psz[nIndex++]-'0') * 100;
		m_date.m_nYear += (psz[nIndex++]-'0') * 10;
		m_date.m_nYear += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_date.m_nMonth = (psz[nIndex++]-'0') * 10;
		m_date.m_nMonth += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_date.m_nDay = (psz[nIndex++]-'0') * 10;
		m_date.m_nDay += (psz[nIndex++]-'0');

		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;

		m_time.m_nHour = (psz[nIndex++]-'0') * 10;
		m_time.m_nHour += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_time.m_nMinute = (psz[nIndex++]-'0') * 10;
		m_time.m_nMinute += (psz[nIndex++]-'0');
		if(psz[nIndex] < '0' || psz[nIndex] > '9')
			nIndex++;
		m_time.m_nSecond = (psz[nIndex++]-'0') * 10;
		m_time.m_nSecond += (psz[nIndex++]-'0');
		return	m_date.m_nYear >= 1900 && m_date.m_nYear <= 2100 && m_date.m_nMonth >= 1 && m_date.m_nMonth <= 12 && m_date.m_nDay >= 1 && m_date.m_nDay <= 31 &&
				m_time.m_nHour >= 0 && m_time.m_nHour <= 23 && m_time.m_nMinute >= 0 && m_time.m_nMinute <= 59 && m_time.m_nSecond >= 0 && m_time.m_nSecond <= 59;
	}
};

#endif // __DateTime_h__
