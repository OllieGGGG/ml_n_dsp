#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#ifdef LINUX_MMAP
#include <sys/mman.h>
#endif

#define KB(size) ((size_t)size << 10)
#define MB(size) ((size_t)size << 20)
#define GB(size) ((size_t)size << 30)

struct Arena
{
    void *buf;
    size_t curr_sz;
    size_t buf_sz;
};

#ifdef LINUX_MMAP
int arenaInit(struct Arena *arena, size_t sz);
int arenaRelease(struct Arena *arena);
#else
void arenaInit(struct Arena *arena, void *backing_buf, size_t sz);
void arenaRelease(struct Arena *arena);
#endif

void *arenaPush(struct Arena *arena, size_t sz, size_t align);
void *arenaPushZero(struct Arena *arena, size_t sz, size_t align);

void arenaPop(struct Arena *arena, size_t sz);
size_t arenaGetSize(struct Arena *arena);
void arenaClear(struct Arena *arena);

#define arenaPushObj(arena, obj_type) arenaPush(arena, sizeof(obj_type), 0);
#define arenaPushObjZero(arena, obj_type) arenaPushZero(arena, sizeof(obj_type), 0);

#ifdef LINUX_MMAP
int arenaInit(struct Arena *arena, size_t sz)
{
    arena->buf = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (arena->buf == NULL)
    {
        return -1;
    }
    arena->curr_sz = 0;
    arena->buf_sz = sz;
    return 0;
}

int arenaRelease(struct Arena *arena)
{
    return munmap(arena->buf, arena->buf_sz); 
}
#else
void arenaInit(struct Arena *arena, void *backing_buf, size_t buf_sz)
{
    arena->buf = backing_buf;
    arena->curr_sz = 0;
    arena->buf_sz = buf_sz;
}

void arenaRelease(struct Arena *arena) {}
#endif

static bool isPowerOfTwo(size_t val)
{
    return (val & (val - 1)) == 0;
}

static uintptr_t alignForwardOffset(uintptr_t addr, size_t align)
{
    if (align == 0) return 0;

    assert(isPowerOfTwo(align));
    uintptr_t b, modulo;
    b = addr;
    
    // Only works if align is power of two
    modulo = b & (align - 1);
    return (modulo != 0) ? (align - modulo) : 0; 
}

void *arenaPush(struct Arena *arena, size_t sz, size_t align)
{
    uintptr_t curr_arena_addr = (uintptr_t)arena->buf + (uintptr_t)arena->curr_sz;
    uintptr_t offset = alignForwardOffset(curr_arena_addr, align);

    if (arena->curr_sz + offset + sz > arena->buf_sz) return NULL;
    arena->curr_sz += offset + sz;

    return (void *)(curr_arena_addr + offset);
}

void *arenaPushZero(struct Arena *arena, size_t sz, size_t align)
{
    void *addr_before_push = (unsigned char *)arena->buf + arena->curr_sz;
    void *addr_after_push = arenaPush(arena, sz, align);

    if (addr_after_push == NULL) return NULL;
    memset(addr_before_push, 0, sz);

    return addr_before_push;
}

void arenaPop(struct Arena *arena, size_t sz)
{
    arena->curr_sz -= (sz > arena->curr_sz) ? 0 : sz;
}

size_t arenaGetSize(struct Arena *arena)
{
    return arena->curr_sz;
}

void arenaClear(struct Arena *arena)
{
    arena->curr_sz = 0;
}
