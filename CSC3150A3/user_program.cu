﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  for (int i = 0; i < 2; i++)
    vm_write(vm, i, input[i]);

  //for (int i = input_size - 1; i >= input_size - 32769; i--) {
	 // uchar value = vm_read(vm, i);
	 // printf("val = %c\n", value);
  //}
    

  //for (int i = 0; i < 64; i++) {
	 // uchar value = vm_read(vm, i);
	 // //printf("val = %s\n", value);
  //}
	  
		

  //vm_snapshot(vm, results, 0, input_size);
}
