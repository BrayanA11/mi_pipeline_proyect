#setup.py
from setuptools import setup, find_packages

setup(
    name="mi_pipeline",
    version="0.1",
    description="Pipeline para detección de enfermedades en hojas",
    author="Brayan",
    author_email="brayanaquinot@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "mi_pipeline": ["swin_b_train_num0_fp16.pth"],
    },
    install_requires=[
        "torch>=1.0",
        "torchvision>=0.10",
        "Pillow>=8.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
