﻿#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"
#include <iostream>
#include <list>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */

	printf("vm_read\n");
	int page_offset = addr % 32;
	int page_num = addr / 32;
	u32 current = vm->invert_page_table[page_num];
	printf("frame = %ld\n", current);
	int frame_num;
	int phy_addr; 
	if (current = 0x800000000) {
		*vm->pagefault_num_ptr += 1; // add one more page fault
	}
	else {
		printf("read frame num\n");
		frame_num = current; 
		phy_addr = frame_num * vm->INVERT_PAGE_TABLE_SIZE + page_offset; 
		printf("value = %c\n", vm->buffer[phy_addr]);
		return vm->buffer[phy_addr];
	}
	
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
	int page_offset = addr % 32;
	int page_num = addr / 32;
	u32 current = vm->invert_page_table[page_num];  // get current frame number
	int frame_num;
	u32 phy_addr;
	printf("curr frame = %ld\n", current);
	if (current == 0x80000000) {
		*vm->pagefault_num_ptr += 1; // add one more page fault
		/*printf("fault num = %d\n", vm->pagefault_num_ptr);*/
		frame_num = page_num; 
		phy_addr = frame_num * vm->PAGESIZE + page_offset;
		vm->buffer[phy_addr] = value;
		printf("write %c\n", *(vm->buffer + phy_addr));
		vm->invert_page_table[page_num] = frame_num;
	}
	else {
		frame_num = current;
		phy_addr = frame_num * vm->PAGESIZE + page_offset;
		vm->buffer[phy_addr] = value;
		printf("write %c\n", *(vm->buffer + phy_addr));
		vm->invert_page_table[page_num] = frame_num;

	}
	//printf("offset = %ld\n", page_offset);
	//printf("page_num = %ld\n", page_num);
	//printf("current phy_addr = %08" PRIx32 "\n", current);


  


}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
	
}

