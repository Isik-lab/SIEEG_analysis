/* Minimal shim to provide iJIT symbols expected by some Intel-linked libs.
   This defines no-op implementations so libtorch_cpu.so can resolve the symbols.
   Build: gcc -shared -fPIC -o scripts/libijitshim.so scripts/ijit_shim.c
   Then run Python with LD_PRELOAD=/full/path/to/scripts/libijitshim.so
   This is a workaround; a proper fix is to install the Intel VTune/ITT library
   that provides these symbols or use a PyTorch build that doesn't reference them.
*/
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Provide a no-op notify function. Signature is approximated to C linkage.
   Real VTune functions have specific signatures; this shim returns safe defaults.
*/
void iJIT_NotifyEvent(void *event, void *data) { (void)event; (void)data; }

int iJIT_IsProfilingActive(void) { return 0; }

unsigned long long iJIT_GetNewMethodID(void) { return 0ULL; }

#ifdef __cplusplus
}
#endif
