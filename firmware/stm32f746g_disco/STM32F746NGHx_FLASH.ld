/*
 * STM32F746NGH6 (STM32F746G-Discovery): 1 MB flash, 64 KB DTCM + 256 KB SRAM1/2.
 *
 *   DTCM   UART receive ring (.dtcm_bss: never cached, reachable by DMA2) and the stack
 *   RAM    .data / .bss / heap
 * The frame buffers and the networks' activations live in the 8 MB SDRAM at 0xC0000000
 * (fixed addresses in Core/Src/main.c, not managed by the linker).
 */
ENTRY(Reset_Handler)

_estack = ORIGIN(DTCM) + LENGTH(DTCM);
_Min_Heap_Size = 0x1000;
_Min_Stack_Size = 0x4000;

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
  DTCM  (rw)  : ORIGIN = 0x20000000, LENGTH = 64K
  RAM   (rwx) : ORIGIN = 0x20010000, LENGTH = 256K
}

SECTIONS
{
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP (*(.init))
    KEEP (*(.fini))
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM : {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >FLASH
  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >FLASH
  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >FLASH

  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >RAM AT> FLASH

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } >RAM

  ._user_heap :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = ALIGN(8);
  } >RAM

  /* not zeroed by the startup code: the ring is written by the DMA before it is read */
  .dtcm_bss (NOLOAD) :
  {
    . = ALIGN(32);
    *(.dtcm_bss)
    *(.dtcm_bss*)
    . = ALIGN(8);
  } >DTCM

  ._user_stack (NOLOAD) :
  {
    . = ALIGN(8);
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >DTCM

  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
