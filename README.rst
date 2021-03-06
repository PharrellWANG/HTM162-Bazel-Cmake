HTM v16.2 with ResNet for Fast Intra Coding
===========================================

This project is

- standing on the shoulder of *HTM v16.2*.
- building a deep learning powered HTM encoder.
- making use of ResNet.
- making use of **Bazel** building and **CMake** building at the same time.

Memos
-----

All the source codes for LIB and APP of HTM encoder are merged, all inside ``/App/TAppEncoder`` folder.

(see FAQ 6 for a detailed explanation of the merging of the source codes for LIB and APP.)

How to compile with SSE4.2 and AVX optimizations using Bazel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use below flags when compiling binary:

.. code-block:: bash

    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --config=cuda -k //PATH/TO/PACKAGE:NAME_OF_PACKAGE


E.G., For our **TAppClassifier** and **TAppEncoder**:

Using GPU
^^^^^^^^^
.. code-block:: bash

    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --config=cuda -k //HTM162/App/TAppClassifier/...
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --config=cuda -k //HTM162/App/TAppEncoder/...

This will make the running speed of your binary the fastest ever(benefiting from avx, sse4.2 offered by your CPU,
AND parallel computing offered by GPU).

.. warning:: If you don't have any parallel computing when doing predictions, then the GPU will help nothing even you
                compiled your binary with GPU support. (E.G., when doing the same non-parallel computing,
                CPU of *Intel core i7* can be faster than GPU of *NVIDIA GTX980*. More specific, running 12288
                predictions for 8x8 size images on Intel core i7 can be 10 seconds faster than on GTX980.)
                Detailed example please refer to:
                http://fast-depth-coding.readthedocs.io/en/latest/tf-speed.html#conclusions

Using Only CPU
^^^^^^^^^^^^^^
.. code-block:: bash

    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 -k //HTM162/App/TAppClassifier/...
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 -k //HTM162/App/TAppEncoder/...

This will make the running speed of your binary faster (benefiting from avx, sse4.2 offered by your CPU).


Branches
--------

- *pharrell_dev_001*: the ``session->Run`` is performed for each block. It is the initial experimental design.

- *pha_dev_002*: the ``session->Run`` is performed for each frame for saving prediction time using tensorflow.


Contact
-------
Pharrell.zx: wzxnuaa@gmail.com


FAQs
----

1. What is HTM?

    The reference software for MV- and 3D-HEVC.

2. What is `Bazel <https://bazel.build/>`_ ?

    {Fast, Correct} - Choose two

    Build and test software of any size, quickly and reliably.

    Most importantly, Tensorflow itself is using *Bazel*. And if you want to make your own binary self-contained,
    you have to put your source code inside tensorflow package hierarchy.


3. What is `CMake <https://cmake.org/>`_ ?

    Build, Test and Package Your Software With CMake.

    Most importantly, if you do not want to use Bazel while you want to use Tensorflow C++ APIs, you have to
    follow this project by using CMake:
    `Build shared library for using the TensorFlow C++ library <https://github.com/FloopCZ/tensorflow_cc>`_

4. Why do you want to integrate ResNet into HTM?

    Want to use neural network to do fast intra mode prediction.

5. Why do you want to use both Bazel and CMake for compiling the HTM encoder binary?

    By enabling CMake with Bazel for the same codebase, we can enjoy two important benefits:

    - *Perfect IDE support* which can help the development [offered by CMake].

    - *Self-Contained binary* which is executable anywhere as long as you are on the same OS [offered by Bazel].


    Below are detailed explanation:

    Here is a link to an open-source project,

    `Build shared library for using the TensorFlow C++ library <https://github.com/FloopCZ/tensorflow_cc>`_

    The above link is pointing to an open-source project which makes possible the usage of
    the `TensorFlow C++ library <https://www.tensorflow.org/api_docs/cc/>`_ from the outside
    of the TensorFlow source code folders and without the use of the `Bazel <https://bazel.build/>`_ build system.

    If you use the shared lib method enabled by the project pointed to by the link above, the binary compiled
    will only be functional on your desktop/laptop which has the shared lib installed.

    But If you are able to make use of `Bazel <https://bazel.build/>`_ for building your c++ binary, your binary would
    be self-contained and executable anywhere (as long as the OS is the same as where the binary is built).

    And we want a binary which is universal for every one to evaluate the quality.

    Well, then why still CMake? Because CMake has a much nicer IDE support. With IDE support you can do much more
    than with a text editor (such as convenient debugging).

    Bazel doesn't have any official IDE support so far. Only a few plugins which are not always functioning well.

    Hence, by enabling CMake with Bazel for the same codebase, we can enjoy both the benefits:

    - *Perfect IDE support* which can help the development [offered by CMake].

    - *Self-Contained binary* which is executable anywhere as long as you are on the same OS [offered by Bazel].

6. Why you merged the libraries such as ``TAppCommon``, ``TLibCommon``, ``TLibRenderer`` and ``libmd5`` etc., into a single folder?

    Because without doing this, you won't be able to use Bazel. Bazel doesn't allow the cycle dependency issue.
    E.g., ``TLibCommon`` is the dependency of ``TAppCommon``, and vice versa. This is introducing a cycle dependency
    issue to Bazel. And Bazel will not allow you to compile your binary before you solve this issue. For solving this
    cycle dependency issue, we have to merge the libs together.

7. **Build** vs **Compile**

    "Building" is a fairly general term, and it can refer to anything that is needed to go
    from editable source material (source code, scripts, raw data files, etc.) to a shippable
    software product. Building can (and usually does) involve several steps, such as pre-processing,
    compiling, linking, converting data files, running automated tests, packaging, etc.

    "Compiling" is more specific, and almost invariably refers to a process that takes source code
    as its input, and outputs something runnable, typically machine code for either a physical or virtual
    machine, or source code in a different language.

    **Compiling** is a sub-set of **Building**.

    We can say that, after building (usually happens in the terminal, not in the IDE), you get a
    shippable binary product; while after compiling (such as
    in the IDE, after your updates to the codes, you compile it for running/debugging, that is to say,
    it usually happens in the IDE, such as Visual Studio), you get a runnable binary, which is not usually
    termed as ``a shippable product``. (E.G, in out case, when we use CMake
    to build our binary, it is not shippable since it heavily depends on the shared lib which will only be linked
    to the binary during runtime. Other machines will not have such shared libs. Hence it is not shippable; BUT,
    if we use Bazel, all the source codes related to the project are built into a single binary. No dependency to
    extra shared lib. Hence the building results will be a shippable product.)

