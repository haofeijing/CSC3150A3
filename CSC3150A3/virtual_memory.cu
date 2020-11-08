#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"
#include <iostream>
#include <list>
#include <queue>


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

	// printf("vm_read\n");
	int page_offset = addr % 32;
	int page_num = addr / 32;

	bool hit = false;
	int hit_slot = -1;
	for (int i = vm->PAGE_ENTRIES-1; i > -1; i--) {
		if (vm->invert_page_table[i] != 0x80000000) {
			if (vm->invert_page_table[i] % (1 << 12) == page_num) {
				hit = true;
				hit_slot = i;
			}
			vm->invert_page_table[i] += (1 << 13);
		}


	}
	int frame_num;
	if (hit) {
		frame_num = vm->invert_page_table[hit_slot + vm->PAGE_ENTRIES];
		u32 phy_addr = frame_num * vm->PAGESIZE + page_offset;
		vm->invert_page_table[hit_slot] = page_num;
		return vm->buffer[phy_addr];
	} else {
		*vm->pagefault_num_ptr += 1; // add page fault;
		// find lru
		int max_time = -1;
		int least_used_slot;
		for (int i = vm->PAGE_ENTRIES-1; i > -1; i--) {
			if (vm->invert_page_table[i] != 0x80000000) {
				int tmp_time = vm->invert_page_table[i] / (1 << 13);
				if (tmp_time > max_time) {
					max_time = tmp_time;
					least_used_slot = i;
				}
			}
		}
		

		frame_num = vm->invert_page_table[least_used_slot + vm->PAGE_ENTRIES];
		int old = vm->invert_page_table[least_used_slot] % (1 << 12);
		for (int i = 0; i < vm->PAGESIZE; i++) {
			// swap out
			vm->storage[old * vm->PAGESIZE + i] = vm->buffer[frame_num * vm->PAGESIZE + i];
			// swap in
			vm->buffer[frame_num * vm->PAGESIZE + i] = vm->storage[page_num * vm->PAGESIZE + i]
		}
		u32 phy_addr = frame_num * vm->PAGESIZE + page_offset;
		vm->invert_page_table[least_used_slot] = page_num;
		return vm->buffer[phy_addr];
	
	}

}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
	int page_offset = addr % 32;
	int page_num = addr / 32;
	int empty_slot = -1;
	bool hit = false;
	int hit_slot = -1;
	// check hit or not
	for (int i = vm->PAGE_ENTRIES-1; i > -1; i--) {
		if (vm->invert_page_table[i] != 0x80000000) {
			if (vm->invert_page_table[i] % (1 << 12) == page_num) {
				hit = true;
				hit_slot = i;
			}
			vm->invert_page_table[i] += (1 << 13);
		}

		
	}
	int frame_num;
	if (hit) {
		frame_num = vm->invert_page_table[hit_slot + vm->PAGE_ENTRIES];
		u32 phy_addr = frame_num * vm->PAGESIZE + page_offset;
		vm->buffer[phy_addr] = value;
		vm->invert_page_table[hit_slot] = page_num;
		
	} else {
		*vm->pagefault_num_ptr += 1; // add page fault;
		// find empty slot
		for (int i = vm->PAGE_ENTRIES-1; i > -1; i--) {
			if (vm->invert_page_table[i] == 0x80000000) {
				empty_slot = i;
			}
		}
		if (empty_slot == -1) {
			// no empty, go to lru
			int max_time = -1;
			int least_used_slot;
			for (int i = vm->PAGE_ENTRIES-1; i > -1; i--) {
				if (vm->invert_page_table[i] != 0x80000000) {
					int tmp_time = vm->invert_page_table[i] / (1 << 13);
					if (tmp_time > max_time) {
						max_time = tmp_time;
						least_used_slot = i;
					}
				}
			}
			

			frame_num = vm->invert_page_table[least_used_slot + vm->PAGE_ENTRIES];
			for (int i = 0; i < vm->PAGESIZE; i++) {
				vm->storage[page_num * vm->PAGESIZE + i] = vm->buffer[frame_num * vm->PAGESIZE + i];
			}
			u32 phy_addr = frame_num * vm->PAGESIZE + page_offset;
			vm->buffer[phy_addr] = value;
			vm->invert_page_table[least_used_slot] = page_num;
		} else {
			// empty exists, write in value in page_number
			vm->invert_page_table[empty_slot] = page_num;
			frame_num = vm->invert_page_table[empty_slot + vm->PAGE_ENTRIES];
			u32 phy_addr = frame_num * vm->PAGESIZE + page_offset;
			vm->buffer[phy_addr] = value;
			
		}
	}
	printf("write: %c\n", value);

	

  


}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
	for (int i = offset; i < input_size; i++) {
		results[i] = vm_read(vm, i);
		printf("snapshot = %c\n", results[i]);
	}
}

