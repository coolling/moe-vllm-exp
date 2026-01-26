#define _GNU_SOURCE
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <sched.h>  // for CPU_SET, cpu_set_t

// ========================
// ZeroCopyBytes: 自定义类型（零拷贝 bytes）
// ========================
typedef struct {
    PyObject_HEAD
    char* data;
    Py_ssize_t size;
} ZeroCopyBytesObject;

static int
zcbo_getbuffer(ZeroCopyBytesObject *self, Py_buffer *view, int flags)
{
    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }
    view->obj = (PyObject*)self;
    view->buf = self->data;
    view->len = self->size;
    view->readonly = 0;  // 允许 PyTorch 写入（消除警告）
    view->itemsize = 1;
    view->format = "B";
    view->ndim = 1;
    view->shape = NULL;
    view->strides = NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    Py_INCREF(self);
    return 0;
}

static void
zcbo_releasebuffer(ZeroCopyBytesObject *self, Py_buffer *view)
{
    // Nothing to do; memory freed in dealloc
}

static PyBufferProcs zcbo_as_buffer = {
    (getbufferproc)zcbo_getbuffer,
    (releasebufferproc)zcbo_releasebuffer
};

static void
zcbo_dealloc(ZeroCopyBytesObject *self)
{
    if (self->data) {
        free(self->data);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject ZeroCopyBytesType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "safetensors_reader.ZeroCopyBytes",
    .tp_doc = "Zero-copy bytes wrapper that owns its buffer",
    .tp_basicsize = sizeof(ZeroCopyBytesObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)zcbo_dealloc,
    .tp_as_buffer = &zcbo_as_buffer,
};

static PyObject*
new_zerocopy_bytes(char* data, Py_ssize_t size)
{
    ZeroCopyBytesObject *obj = (ZeroCopyBytesObject*)ZeroCopyBytesType.tp_alloc(&ZeroCopyBytesType, 0);
    if (obj == NULL) {
        free(data);
        return NULL;
    }
    obj->data = data;
    obj->size = size;
    return (PyObject*)obj;
}

// ========================
// Worker for single contiguous buffer (with CPU affinity)
// ========================
typedef struct {
    int fd;
    char* buffer_base;
    off_t file_start;
    size_t start_idx;
    size_t count;
    bool success;
    int cpu_id;  // ← 新增：目标 CPU 核心编号
} single_buffer_worker_arg_t;

static void*
single_buffer_worker(void* arg)
{
    single_buffer_worker_arg_t* a = (single_buffer_worker_arg_t*)arg;

    // === 绑定当前线程到指定 CPU 核心 ===
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(a->cpu_id, &cpuset);
    // 忽略返回值（best-effort，失败也不影响功能）
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    // =================================

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    ssize_t n = pread(a->fd,
                      a->buffer_base + a->start_idx,
                      a->count,
                      a->file_start + a->start_idx);
    a->success = (n == (ssize_t)a->count);

    clock_gettime(CLOCK_MONOTONIC, &end);
    long long total_elapsed_us = (end.tv_sec - start.tv_sec) * 1000000LL +
                                 (end.tv_nsec - start.tv_nsec) / 1000;
    fprintf(stderr, "[Thread on CPU %d] read time: %lld us\n", a->cpu_id, total_elapsed_us);
    return NULL;
}

// ========================
// Main Function: 返回单块 ZeroCopyBytes
// ========================
static PyObject* parallel_read_and_parse(PyObject* self, PyObject* args) {
    const char* file_path;
    const char* weight_name;
    int max_workers = -1;
    int chunk_size_mb = 1;

    if (!PyArg_ParseTuple(args, "ss|ii", &file_path, &weight_name, &max_workers, &chunk_size_mb)) {
        return NULL;
    }

    if (chunk_size_mb <= 0) {
        PyErr_SetString(PyExc_ValueError, "chunk_size_mb must be positive");
        return NULL;
    }

    // --- Step 1: Parse safetensors header ---
    int fd = open(file_path, O_RDONLY);
    if (fd == -1) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, file_path);
        return NULL;
    }

    unsigned char header_len_buf[8];
    if (read(fd, header_len_buf, 8) != 8) {
        close(fd);
        PyErr_SetString(PyExc_ValueError, "Failed to read 8-byte header length");
        return NULL;
    }

    uint64_t header_len =
        ((uint64_t)header_len_buf[0]      ) |
        ((uint64_t)header_len_buf[1] <<  8) |
        ((uint64_t)header_len_buf[2] << 16) |
        ((uint64_t)header_len_buf[3] << 24) |
        ((uint64_t)header_len_buf[4] << 32) |
        ((uint64_t)header_len_buf[5] << 40) |
        ((uint64_t)header_len_buf[6] << 48) |
        ((uint64_t)header_len_buf[7] << 56);

    char* header_json = malloc(header_len);
    if (!header_json) {
        close(fd);
        PyErr_NoMemory();
        return NULL;
    }

    if (read(fd, header_json, header_len) != (ssize_t)header_len) {
        free(header_json);
        close(fd);
        PyErr_SetString(PyExc_ValueError, "Failed to read header JSON");
        return NULL;
    }
    close(fd);

    PyObject* json_mod = PyImport_ImportModule("json");
    if (!json_mod) {
        free(header_json);
        return NULL;
    }

    PyObject* loads = PyObject_GetAttrString(json_mod, "loads");
    Py_DECREF(json_mod);
    if (!loads) {
        free(header_json);
        return NULL;
    }

    PyObject* header_str = PyUnicode_FromStringAndSize(header_json, header_len);
    free(header_json);
    if (!header_str) {
        Py_DECREF(loads);
        return NULL;
    }

    PyObject* header = PyObject_CallFunctionObjArgs(loads, header_str, NULL);
    Py_DECREF(loads);
    Py_DECREF(header_str);
    if (!header) return NULL;

    if (!PyDict_Check(header)) {
        Py_DECREF(header);
        PyErr_SetString(PyExc_ValueError, "Header is not a dict");
        return NULL;
    }

    PyObject* weight_info = PyDict_GetItemString(header, weight_name);
    if (!weight_info) {
        Py_DECREF(header);
        PyErr_Format(PyExc_KeyError, "Weight '%s' not found", weight_name);
        return NULL;
    }

    PyObject* offsets = PyDict_GetItemString(weight_info, "data_offsets");
    if (!offsets || !PyList_Check(offsets) || PyList_Size(offsets) != 2) {
        Py_DECREF(header);
        PyErr_SetString(PyExc_ValueError, "Invalid data_offsets");
        return NULL;
    }

    long long offset0 = PyLong_AsLongLong(PyList_GetItem(offsets, 0));
    long long offset1 = PyLong_AsLongLong(PyList_GetItem(offsets, 1));
    if (offset0 == -1 && PyErr_Occurred()) { Py_DECREF(header); return NULL; }
    if (offset1 == -1 && PyErr_Occurred()) { Py_DECREF(header); return NULL; }
    if (offset0 < 0 || offset1 < offset0) {
        Py_DECREF(header);
        PyErr_SetString(PyExc_ValueError, "Invalid data_offsets range");
        return NULL;
    }

    off_t data_start_pos = 8 + (off_t)header_len;
    off_t tensor_start = data_start_pos + offset0;
    size_t tensor_size = (size_t)(offset1 - offset0);
    Py_DECREF(header);

    if (tensor_size == 0) {
        char* empty = malloc(0);
        return new_zerocopy_bytes(empty, 0);
    }

    // --- Step 2: Determine number of workers ---
    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count <= 0) cpu_count = 4;

    int workers = (max_workers <= 0) ? (int)(cpu_count * 2) : max_workers;
    if (workers > 8) workers = 8;
    if (workers <= 0) workers = 1;
    if ((size_t)workers > tensor_size) workers = (int)tensor_size;

    // Allocate ONE contiguous buffer
    char* full_buffer = malloc(tensor_size);
    if (!full_buffer) {
        PyErr_NoMemory();
        return NULL;
    }

    int main_fd = open(file_path, O_RDONLY);
    if (main_fd == -1) {
        free(full_buffer);
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, file_path);
        return NULL;
    }

    if (flock(main_fd, LOCK_SH) != 0) {
        close(main_fd);
        free(full_buffer);
        PyErr_SetFromErrno(PyExc_OSError);
        return NULL;
    }

    size_t chunk_size = (tensor_size + workers - 1) / workers;

    single_buffer_worker_arg_t* wargs = calloc(workers, sizeof(single_buffer_worker_arg_t));
    pthread_t* threads = calloc(workers, sizeof(pthread_t));
    if (!wargs || !threads) {
        flock(main_fd, LOCK_UN);
        close(main_fd);
        free(full_buffer);
        free(wargs);
        free(threads);
        PyErr_NoMemory();
        return NULL;
    }

    // Assign ranges AND CPU IDs
    for (int w = 0; w < workers; w++) {
        size_t start = w * chunk_size;
        if (start >= tensor_size) {
            wargs[w].count = 0;
            continue;
        }
        size_t count = (w == workers - 1) ? (tensor_size - start) : chunk_size;
        wargs[w].fd = main_fd;
        wargs[w].buffer_base = full_buffer;
        wargs[w].file_start = tensor_start;
        wargs[w].start_idx = start;
        wargs[w].count = count;
        wargs[w].success = true;
        wargs[w].cpu_id = w % cpu_count;  // ← 轮询分配 CPU 核心
    }

    Py_BEGIN_ALLOW_THREADS

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int w = 0; w < workers; w++) {
        if (wargs[w].count == 0) continue;
        if (pthread_create(&threads[w], NULL, single_buffer_worker, &wargs[w]) != 0) {
            wargs[w].success = false;
        }
    }

    for (int w = 0; w < workers; w++) {
        if (wargs[w].count > 0) {
            pthread_join(threads[w], NULL);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    long long total_elapsed_us = (end.tv_sec - start.tv_sec) * 1000000LL +
                                 (end.tv_nsec - start.tv_nsec) / 1000;
    fprintf(stderr, "[Main] Total parallel read time: %lld us\n", total_elapsed_us);

    bool any_fail = false;
    for (int w = 0; w < workers; w++) {
        if (!wargs[w].success) {
            any_fail = true;
            break;
        }
    }

    flock(main_fd, LOCK_UN);
    close(main_fd);
    free(wargs);
    free(threads);

    if (any_fail) {
        free(full_buffer);
        PyErr_SetString(PyExc_RuntimeError, "Failed to read part of the tensor from disk");
        return NULL;
    }

    Py_END_ALLOW_THREADS

    return new_zerocopy_bytes(full_buffer, tensor_size);
}

// ========================
// Module Definition
// ========================
static PyMethodDef methods[] = {
    {"parallel_read_and_parse", parallel_read_and_parse, METH_VARARGS,
     "Read a safetensors weight as a single ZeroCopyBytes object (zero-copy)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "safetensors_reader",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_safetensors_reader(void)
{
    if (PyType_Ready(&ZeroCopyBytesType) < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&module);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&ZeroCopyBytesType);
    if (PyModule_AddObject(m, "ZeroCopyBytes", (PyObject *)&ZeroCopyBytesType) < 0) {
        Py_DECREF(&ZeroCopyBytesType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}