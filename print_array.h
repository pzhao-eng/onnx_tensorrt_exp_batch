/*************************************************************************
      > File Name: print_array.h
      > Author: zhaopeng
      > Mail: zhaopeng_chem@163.com
      > Created Time: Thu 14 Apr 2022 12:55:51 PM CST
 ************************************************************************/

#ifndef _PRINT_ARRAY_H
#define _PRINT_ARRAY_H
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdio.h>

using namespace std;

void print_device(void *arr, int size);
#endif
