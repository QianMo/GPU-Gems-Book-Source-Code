#pragma once

typedef unsigned short*(*GeneratePtr)(int);

unsigned short* dumbell(int _size);
unsigned short* pyramid(int _size);
unsigned short* cube(int _size);
unsigned short* random(int _size);
