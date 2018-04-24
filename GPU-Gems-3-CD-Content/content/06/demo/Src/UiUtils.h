#ifndef __UI_UTILS_H
#define __UI_UTILS_H

#include "Cfg.h"
#include "DXUTgui.h"

inline int toSlider(int i)
{
	return i;
}

inline int toSlider(float f)
{
	return static_cast<int>(f * (1.0f/0.01f));
}

template <typename R>
inline R fromSlider(int sliderValue)
{
	return sliderValue;
}

template <>
inline float fromSlider(int sliderValue)
{
	return static_cast<float>(sliderValue) * 0.01f;
}

template <typename SliderType>
inline SliderType updateSlider(CDXUTSlider* slider, CDXUTStatic* caption, LPCWSTR captionText, SliderType& value, bool read = true)
{
	assert(slider);

	WCHAR sz[100];
	if(!read)
		slider->SetValue(toSlider(value));

	value = fromSlider<SliderType>(slider->GetValue());
	StringCchPrintf(sz, 100, captionText, value); 
	caption->SetText(sz);
	return value;
}

inline void updateComboBox(CDXUTComboBox* combo, size_t index)
{
	assert(combo);
	combo->SetSelectedByIndex(static_cast<UINT>(index));
}

#endif
