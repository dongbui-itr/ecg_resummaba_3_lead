/* newlib stubs: nothing here uses files; snprintf only pulls these in through its
 * reentrancy structure (they would otherwise come from libnosys with link warnings). */
#include <errno.h>
#include <sys/stat.h>

int _close(int fd) { (void)fd; errno = EBADF; return -1; }
int _lseek(int fd, int off, int whence) { (void)fd; (void)off; (void)whence; errno = ESPIPE; return -1; }
int _read(int fd, char *buf, int n) { (void)fd; (void)buf; (void)n; errno = EBADF; return -1; }
int _write(int fd, const char *buf, int n) { (void)fd; (void)buf; (void)n; errno = EBADF; return -1; }
