from setuptools import setup, Extension

module = Extension(
    'safetensors_reader',
    sources=['safetensors_reader.c'],
    extra_compile_args=['-O3', '-std=c99', '-pthread'],
    extra_link_args=['-pthread']
)

setup(
    name='safetensors_reader',
    ext_modules=[module]
)