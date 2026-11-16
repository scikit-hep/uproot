# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

"""
This module defines behaviors, which are mix-in classes that provide a high-level interface
to objects read from ROOT files.

Behaviors do not have a required base class, and they may be used with
Awkward Array's ``ak.behavior``.

To add a behavior for a ROOT class:

1. Translate the ROOT class name from C++ to Python with
   :doc:`uproot.model.classname_encode`. For example,
   ``"ROOT::RThing"`` becomes ``"Model_ROOT_3a3a_RThing"``.
2. Create a submodule of ``uproot.behaviors`` without
   the ``"Model_"`` prefix. For example, ``"ROOT_3a3a_RThing"``.
3. Include a class in that submodule with the fully encoded
   name. For example, ``"Model_ROOT_3a3a_RThing"``.

When Uproot reads an instance of the class, it would always create a
deserialization model if one is not predefined. But if a behavior with the
appropriate name exist, the new class will inherit from the behavior, giving
the newly created object specialized methods and properties.

See also :doc:`uproot.models`.
"""
