Beginners’ tutorial
===================

PyLlamas is a Python code that enables to calculate the reflection spectrum of multilayer stacks in a few simple steps and to export the results in Python-compatible or in MATLAB format. Not everyone is experienced in programming in general or in Python in particular, and in this tutorial we explain in a less formal way what you need to do from the very start to model simple or complex stacks without any prior Python knowledge.

If you have never used Python and are not very comfortable with programming in general, the most difficult step will be to install everything. Then, running a code and changing your parameters will be relatively easy.

How to install Python
---------------------

A code is a list of instructions written in a given programming language. Your computer will interpret the instructions and perform tasks accordingly. In order for your computer to understand instructions written in Python, you need to install the `Python <http://www.python.org/>`_ interpreter from its official website: pick the "Download" menu, pick your operating system (for example Windows) and download the executable file (for example "Windows x86-64 executable installer") for the Python version that you want.

There are several versions of Python: roughly speaking, each version corresponds to different sets of instructions that can be written by the programmer and understood by the computer. The first version of this code was written with Python 3.6. You should download Python 3.6.X where X can be any number. StackMat will very likely work with other Python versions that start with 3.X.X but this is not guaranteed.

..
    TODO check Python 3.8

You also need to use an Integrated Development Interface (IDE). This is a software that will enable you to open a Python code, modify it (to change your parameters) and run it. An IDE comes with Python and is called IDLE: when you double-click on a .py file, IDLE will open it and you just have to press "Run" on the top-menu to launch the code. However, IDLE is not very handy to use and you may want to switch to `JetBrain’s PyCharm <https://www.jetbrains.com/pycharm/>`_ (there is a free version, or you can download the paid version for free if you are a student) or `Anaconda’s Spyder <https://www.anaconda.com/>`_.

Your IDE needs to know where the Python interpreter is located on your computer. This is automatic for IDLE. If you install another IDE, you might need to find tutorials on their websites to help you configure it.

How to install packages that PyLlama needs
-------------------------------------------

StackMat uses functions that are implemented in common Python packages (for example, the cosine function comes with the package NumPy). To use StackMat, you need to install the following packages:

- NumPy version 3.18.2
- ScyPy version 1.2.1
- SymPy version 1.4
- Matplotlib version 3.2.1

Other versions may work: a package does not completely change from version to version.

The steps that you need to follow to install your packages depend on the IDE that you are using. Tutorials are available online to help beginners with the package installation.

- With IDLE (that directly comes with Python), instructions can be found on `Python’s websive <https://packaging.python.org/tutorials/installing-packages/>`_
- With PyCharm, instructions can be found on `Jetbrain’s websive <https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html#>`_. No command-line is required and you can do everything through the PyCharm software, including picking a specific package version.
- With Spyder, instructions can be found on `Anacodna’s websive <https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/>`_

..
    TODO add versions of packages
    TODO remove SymPy altogether

How to install PyLlama
-----------------------

The StackMat code is stored on GitHub. You should go to `its GitHub repository <https://www.google.com>`_ and click on the download button. Then, unzip the archive and place all files in a dedicated folder.

..
    TODO add link to GitHub

There are several files in the archive:

- ``pyllama.py``: the StackMat code is a set of pre-defined sets of instructions (functions) that you can use. You don’t need to modify it and you should not run it. However, your interpreter needs to access this file to use the functions that it contains.
- ``cholesteric.py`` and ``geometry.py``: these codes contain pre-defined instructions that are required to model cholesterics and to display 3D representations. Your interpreter also needs to access these file to use the functions that it contains if you want to model cholesterics.
- a collection of files that are named ``script_XXXX.py``: these are examples of codes that you can run. These scripts use functions from ``stackmat.py`` in order to model different examples of stacks, for example to plot a spectrum. You can write your script from scratch or you can find one that resembles the stack that you want to model and edit it.

How to modify and run a script
------------------------------



How to write a script for an arbitrary stack
--------------------------------------------


