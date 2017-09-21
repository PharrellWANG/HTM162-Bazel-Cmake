HTM v16.2 with ResNet for Fast Intra Coding
===========================================

This project is standing on the shoulder of *HTM v16.2*.

This project is making use of **Bazel** building and **CMake** building at the same time.

FAQs for every one
------------------

1. What is HTM?

    The reference software for MV- and 3D-HEVC.

2. What is `Bazel <https://bazel.build/>`_ ?

    {Fast, Correct} - Choose two

    Build and test software of any size, quickly and reliably.


3. What is `CMake <https://cmake.org/>`_ ?

    Build, Test and Package Your Software With CMake.

4. Why do you want to integrate ResNet into HTM?

    Speed matters.

5. Why do you want to use both Bazel and CMake for compiling the HTM encoder binary?

    `Build shared library for using the TensorFlow C++ library <https://github.com/FloopCZ/tensorflow_cc>`_

    The above link is pointing to an open-source project which makes possible the usage of
    the `TensorFlow C++ library <https://www.tensorflow.org/api_docs/cc/>`_ from the outside
    of the TensorFlow source code folders and without the use of the `Bazel <https://bazel.build/>`_ build system.

    If you use the shared lib method enabled by the project pointed to by the link above, the binary compiled
    will only be functional on your desktop/laptop which has the shared lib installed.

    But If you are able to make use of `Bazel <https://bazel.build/>`_ for building your c++ binary, your binary would
    be self-contained and executable anywhere (as long as the OS is the same as where the binary is built).

    And we want a binary which is universal for every one to evaluate the quality.

    Well, then why still CMake? Because CMake has a much nicer IDE support. Bazel doesn't have any official IDE
    support so far. Only a few plugins which are not always functioning well.

    By enabling CMake with Bazel for the same codebase, we can enjoy both the benefits:

    - IDE support.

    - Self-Contained binary.

FAQs for programmers
--------------------

1. Why you merged the libraries such as ``TAppCommon``, ``TLibCommon``, ``TLibRenderer`` and ``libmd5`` etc., into a single folder?

    Because without doing this, you won't be able to use Bazel. Bazel doesn't allow the cycle dependency issue.
    E.g., ``TLibCommon`` is the dependency of ``TAppCommon``, and vice versa. This is introducing a cycle dependency
    issue to Bazel. And Bazel will not allow you to compile your binary before you solve this issue. For solving this
    cycle dependency issue, we have to merge the libs together.



