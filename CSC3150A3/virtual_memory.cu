#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"
#include <iostream>
#include <list>
#include <queue>

__device__ linked_list ll;

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
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == page_num) {
			int phy_addr = vm->invert_page_table[i + page_num] * vm->PAGESIZE + page_offset;
			return vm->buffer[phy_addr];
		}
	}
	*vm->pagefault_num_ptr += 1; 
	node * head = ll.head;
	int least_used_slot = head->value;
	ll.head = head->prev;
	ll.head->next = NULL;
	ll.size -= 1;
	
	int old = vm->invert_page_table[least_used_slot];
	int frame_num = vm->invert_page_table[least_used_slot + vm->PAGE_ENTRIES];
	for (int i = 0; i < vm->PAGESIZE; i++) {
		vm->storage[old * vm->PAGESIZE + i] = vm->buffer[frame_num * vm->PAGESIZE + i];
		vm->buffer[frame_num * vm->PAGESIZE + i] = vm->storage[page_num * vm->PAGESIZE + i];
	}
	return vm->buffer[frame_num * vm->PAGESIZE + page_offset];


	
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
	int page_offset = addr % 32;
	int page_num = addr / 32;
	int empty_slot = -1;
	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == 0x80000000) {
			empty_slot = i;
		}
	}
	//printf("empty = %d\n", empty_slot);
	int frame_num;
	if (empty_slot != -1) {
		frame_num = vm->invert_page_table[empty_slot + vm->PAGE_ENTRIES];

		//q.push(empty_slot);
		node tmp;
		tmp.value = empty_slot;
		if (ll.size == 0) {
			ll.tail = &tmp;
			ll.head = &tmp;
		}
		else {
			ll.tail->prev = &tmp;
			tmp.next = ll.tail;
			ll.tail = &tmp;
		}
		ll.size += 1;

		vm->invert_page_table[empty_slot] = page_num;
	}
	else {
		node * head = ll.head;
		int least_used_slot = head->value;
		ll.head = head->prev;
		ll.head->next = NULL;
		ll.size -= 1;


		frame_num = vm->invert_page_table[least_used_slot + vm->PAGE_ENTRIES];
		for (int i = 0; i < vm->PAGESIZE; i++) {
			vm->storage[page_num * vm->PAGESIZE + i] = vm->buffer[frame_num * vm->PAGESIZE + i];
		}
	}
	u32 phy_addr = frame_num * vm->PAGESIZE + page_offset;
	vm->buffer[phy_addr] = value;
	printf("write %c\n", *(vm->buffer + phy_addr));




	//printf("offset = %ld\n", page_offset);
	//printf("page_num = %ld\n", page_num);
	//printf("current phy_addr = %08" PRIx32 "\n", current);
	

  


}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
	for (int i = offset; i < input_size; i++) {
		results[i] = vm_read(vm, i);
	}
}

