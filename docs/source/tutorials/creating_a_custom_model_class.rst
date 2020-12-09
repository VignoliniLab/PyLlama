Creating a custom ``Model`` class
=================================

The ``Model`` class and its children enable the user to construct ``Structures`` automatically from given parameters. The class ``Model`` can be viewed as an abstract class that defines parameters and methods common to all its children; however, it is possible to create an instance of ``Model``: it will have an empty ``Structure``. In the tutorial, we explain how the class ``Model`` is constructed and how the user can write their own child class.

Anatomy of the ``Model`` class
------------------------------

The parameters of the ``Model`` class are:

- ``n_entry``: the refractive index of the stack’s entry isotropic semi-infinite medium
- ``n_exit``: the refractive index of the stack’s exit isotropic semi-infinite medium
- ``wl_nm``: the wavelength in nanometers
- ``theta_in_rad``: the angle of incidence in radians

and they are used to initialise the ``Model`` with the following fields:
::
    def __init__(self, n_entry, n_exit, wl_nm, theta_in_rad):
        self.n_entry = n_entry
        self.n_exit = n_exit
        self.wl = wl_nm
        self.theta_in = theta_in_rad
        theta_out = np.arcsin((n_entry / n_exit) * np.sin(self.theta_in))
        self.k0 = 2 * np.pi / self.wl
        self.Kx = self.n_entry * np.sin(self.theta_in)          # kx = Kx * k0
        self.Ky = 0                                             # ky = Ky * k0
        self.Kz_entry = self.n_entry * np.cos(self.theta_in)    # kz_entry = Kz_entry * k0
        self.Kz_exit = self.n_exit * np.cos(theta_out)          # kz_entry = Kz_entry * k0
        self.structure = self._build_structure_total()

The ``Model``'s ``structure`` is an instance of ``Structure`` created with the function ``_build_structure_total()``:
::
    def _build_structure_total(self):
        entry_space, exit_space = self._build_entry_exit()
        structure = self._build_structure(entry_space, exit_space)
        return structure

This function calls two sub-functions: ``_build_entry_exit()`` that creates the entry and exit isotropic ``HalfSpaces``, and ``_build_structure()`` that creates a ``Structure`` containing the appropriate ``Layers`` to represent the multilayer stack.

The function ``_build_entry_exit()`` simply creates the entry and exit ``HalfSpaces``:
::
    def _build_entry_exit(self):
        epsilon_entry = np.array([[self.n_entry ** 2, 0, 0],
                                  [0, self.n_entry ** 2, 0],
                                  [0, 0, self.n_entry ** 2]])
        epsilon_exit = np.array([[self.n_exit ** 2, 0, 0],
                                 [0, self.n_exit ** 2, 0],
                                 [0, 0, self.n_exit ** 2]])
        entry_space = HalfSpace(epsilon_entry, self.Kx, self.Kz_entry, self.k0, category="isotropic")
        exit_space = HalfSpace(epsilon_exit, self.Kx, self.Kz_exit, self.k0, category="isotropic")
        return entry_space, exit_space

and the function ``_build_structure()`` creates the ``Structure`` representing the multilayer stack, without any ``Layer`` when the class ``Model`` is used:
::
    def _build_structure(self, entry_space, exit_space):
        warnings.warn("The build_function method of the Model class is used.")
        return Structure(entry=entry_space, exit=exit_space, Kx=self.Kx, Ky=self.Ky, Kz_entry=self.Kz_entry, Kz_exit=self.Kz_exit, k0=self.k0, N_periods=1)

When the ``Structure`` has been built, its reflectance can be calculated with ``get_refl_trans()``:
::
    def get_refl_trans(self, circ=False, method="SM"):
        return self.structure.get_refl_trans(circ=circ, method=method)

Creating a custom child
-----------------------

The core functions in the ``Model`` class are the following:

- ``__init__`` to create the ``Model`` instance
- ``_build_entry_exit()`` to create the entry and exit ``HalfSpaces``
- ``_build_structure()`` to create the ``Structure`` with the ``Layers``
- ``_build_structure_total()`` that calls ``_build_entry_exit()`` and ``_build_structure()``
- ``get_refl_trans()`` that calculates the reflectance of the multilayer stack represented by the ``Model``

This constitutes a blueprint for the children classes of ``Model``, such as ``StackModel`` or ``CholestericModel``. A child class of ``Model`` contains functions that can be divided into three categories:

- functions that are implemented in ``Model`` and that the child class inherits
- functions that are implemented in ``Model`` and that are overwritten in the child class
- functions that are specific to the child class

Typically, the user's new child class will be written as follows:
::
    class ChildModel(Model):
        def __init__(self, parameter1, parameter2, parameter3, n_entry, n_exit, wl_nm, theta_in_rad):
            # Initialisation with parameters that are specific to ChildModel:
            self.param1 = parameter1
            self.param2 = parameter2
            self.param3 = parameter3
            # Initialisation with the inherited method:
            # (ChildModel's parameters might be used to recalculate the parent's parameters)
            super().__init__(n_entry, n_exit, wl_nm, theta_in_rad)

        def _build_structure(self, entry_space, exit_space):
            # Create an empty structure between isotropic half spaces
            my_structure = Structure(entry_space, exit_space, self.Kx, self.Ky, self.Kz_entry, self.Kz_exit, self.k0, N_per=1)

            # A custom routine with self.param1, self.param2, self.param3 that creates Layers and adds them to the Structure
            my_structure.add_layers(my_list_of_layers)

            # Return the Structure that contains the custom-made Layers
            return my_structure

When using ``super().__init__`` in the child class (``ChildModel``), it will call the ``_build_structure_total()`` method in the parent class (``Model``), which will then call both the ``_build_structure()`` method of the child (which overrides the parent one), and ``_build_entry_exit()`` of the parent (since it is not overridden by a child version). ``CholestericModel``, ``SlabModel``, ``StackModel`` and ``StackOpticalThicknessModel`` are built this way. They also inherit ``get_refl_trans()`` from ``Model``.

The user simply needs to add their own ``ChildModel`` to the code by using the sample used as an example above with their own chosen parameters, and when calling ``ChildModel.get_refl_trans()``, they will immedietaly benefit from the optical calculations that have been implemented.

Of course, when the user writes a new child class, they may overwrite as many functions as they want, and they may add as many specific functions as they want. For example, ``MixedModel`` overwrites most functions from ``Model``.

``Model`` also contains the function ``copy_as_stack()`` that creates a ``StackModel`` containing the same layers as a given ``Model``. The user will need to overwrite this function too.

Pairing the custom child with ``Spectrum``
------------------------------------------

The ``Spectrum`` class implements the modelling of a multilayer stack over a range of wavelength and provide tools for calculating reflection spectra with the choice of the polarisation basis and for exporting the data. Pairing the user's new child class of ``Model`` with ``Spectrum`` enables the user to have access to such functionalities.

A ``Spectrum`` is defined by the following function:
::
    def __init__(self, wl_nm_list, model_type, model_parameters):
        self.wl_list = wl_nm_list         # list of wavelengths in nm
        self.mo_type = model_type         # name of model
        self.mo_param = model_parameters  # dictionary with model parameters
        self.data = {}                    # empty dictionary for storage

When the user calls the function ``calculate_refl()``, the name of the ``Model`` (``mo_type``) is checked and this triggers the creation of the appropriate ``Model``, from the parameters in the dictionary ``mo_param``. The user needs to add their own ``elif`` case to identify the ``ChildModel`` and handle its parameters correctly. For example, for the following ``ChildModel``'s input parameters:
::
    def __init__(self, parameter1, parameter2, parameter3, n_entry, n_exit, wl_nm, theta_in_rad, default4=value4, default5=value5)

the new ``elif`` case to add to ``Spectrum``’s ``calculate_refl()`` corresponds to:
::
    elif self.mo_type == "ChildModel":
        default_param = dict("default4"=value4, "default5"=value5)
        self.mo_param = {**default_param, **self.mo_param}  # self.mo_param is added to default_param and overwrites the default parameters
        model = ChildModel(self.mo_param["parameter1"],
                           self.mo_param["parameter2"],
                           self.mo_param["parameter3"],
                           self.mo_param["n_entry"],
                           self.mo_param["n_exit"],
                           wl,
                           self.mo_param["theta_in_rad"],
                           self.mo_param["default4"],
                           self.mo_param["default5"])

This says that when ``Spectrum`` is instanciated with the parameter ``mo_type`` equal to the string ``ChildModel``, an instance of ``ChildModel`` will be created with the parameters chosen by the user.

The keys ``"parameter1"``, ``"parameter2"``, etc, can have an arbitrary name, but for clarity it is easier if the keys match the parameter's name in the ``__init__`` function.

Once this is done, the user can create a ``Spectrum`` with ``ChildModel`` as usual, as well as calculate the reflectance and export the spectra:
::
    # Creation of the wavelengths
    wl_nm_list = range(400, 800)

    # Parameters for the ChildModel
    # There are two default parameters: default4 and default5
    # The user sets a value for default5: this overwrites the default value
    # The user doesn't set a value for default4: the default value will be used
    model_type = "ChildModel"
    model_parameters = {"parameter1": my_value_1,
                        "parameter2": my_value_2,
                        "parameter3": my_value_3,
                        "default5": my_value_5,
                        "n_entry": n_entry,
                        "n_exit": n_exit,
                        "theta_in_rad": theta_in_rad}

    # Creation of the periodic stack
    my_spec = Spectrum(wl_nm_list, model_type, model_parameters)

    # The functions of the Spectrum class automatically work
    my_stack_spec.calculate_refl_trans()
    matplotlip.pyplot.plot(wl_nm_list, my_stack_spec.data["R_ps"])
    my_stack_spec.export("my_file_name.mat")

