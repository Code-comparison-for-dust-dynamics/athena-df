"""
  Read Athena++ output data files and fill in empty regions in each level.
  Author: Rixin Li
  Email: rixin.li.astro@gmail.com
"""

import numpy as np
import scipy as sp
import scipy.interpolate as spint
import matplotlib.pyplot as plt
import athena_read as athr
import re
import warnings
import pyridoxine.plt as rxplt

def unique_with_atol(A, atol=1e-7):
    equals = np.isclose(A[:-1], A[1:], atol=atol)
    if equals[-1] is True:
        return (A[:-1])[~equals]
    else:
        return np.hstack([(A[:-1])[~equals], A[-1]])
    
def find_min_restrictable_level(max_level, block_size):
    if max_level == 0:
        return 0
    else:
        for _level in range(max_level-1, -1, -1):  # from 2nd finest to root level
            max_restrict_factor = 2**(max_level - _level)
            for _block_size in block_size:
                if (_block_size != 1 and _block_size % max_restrict_factor != 0):
                    return _level + 1
        return 0
    
def dust_var_check(key_name):
    pattern = r'dust_\d+_((vel|mom|diff_mom|adv_mom|adv_vel)[1-3]|rho|dens)'
    return bool(re.fullmatch(pattern, key_name))

def simplified_comp_check(key_name):
    return bool(re.match(r'[uvM]([1-3]|[xyzr]|phi|theta|[φϕθ])', key_name))

def dust_simplified_check(key_name):
    return bool(re.match(r'rho(d|p)_\d+', key_name) or
                re.match(r'(w|W)([1-3]|[xyzr]|phi|theta|[φϕθ])(_(adv|diff|dif))?_\d+', key_name)
                )

def magnitude_check(key_name):
    return bool(
        key_name in ("Mmag", "umag", "vmag")
        or re.match(r'(W|w)mag(_(adv|diff|dif))?_\d+$', key_name)
    )

class Raw_Data_Restricted:
    
    def __init__(self, filename, q=None, **kwargs):
        
        if "ds_raw" in kwargs.keys():
            ds_raw = kwargs.get("ds_raw")
        else:
            ds_raw = athr.athdf(filename, raw=True, quantities=q, )  # quantities not used if raw=True
        
        self.ds_raw = ds_raw
        self.filename = filename
        self.kwargs = kwargs
        self.warn_derived = kwargs.get("warn_derived", True)
        
        self.t = ds_raw['Time']
        self.var_names = ds_raw['VariableNames']
        self.ds_name = ds_raw['DatasetNames'][0].decode('utf-8')  # it is either b'cons' or b'prim'
        if q is None:  # todo: check if input quantities are included in VariableNames
            self.quantities = [x.decode('ascii', 'replace') for x in self.var_names]
        elif isinstance(q, str):
            self.quantities = [q, ]
        else:
            self.quantities = q
    
        # mostly copied from athena_read.py
        self.block_size = ds_raw['MeshBlockSize']
        self.root_grid_size = ds_raw['RootGridSize']
        self.levels = ds_raw['Levels']
        self.logical_locations = ds_raw['LogicalLocations']
        self.max_level = ds_raw['MaxLevel']  # it's level index, starting from 0
        self.old_num_mb = ds_raw['NumMeshBlocks']
        
        # for now, if ndim = 2, we assume only the first two dimensions are meaningful
        self.ndim = self.get_ndim()
        
        # w/o knowing the minimum level that the finest level can be restricted to, otherwise, we'll running into the following error:
        # "Block boundaries at finest level must be cell boundaries at desired level for subsampling or fast restriction to work" 
        # In other words, restrictions from the finest level to a certain coarse level will fail if:
        #   (A) a meshblock in the finest level is smaller than one cell in that coarse level;
        #   (B) a meshblock in the finest level only covers odd number of cells in that coarse level (even number is needed).
        # Thus, using np.log2 to determine min_restrict-able_level only works if Nx in meshblock is 2^(...); need to check (B) as well.
        #
        #block_refine_limit = np.log2(self.block_size)
        #self.block_refine_limit = int(np.floor(block_refine_limit[block_refine_limit>0].min()))
        self.min_restrictable_level = find_min_restrictable_level(self.max_level, self.block_size)

        # requested quantities the file does not store (e.g. prim names on a combined archive) are
        # derived cell by cell here, so restriction below treats them like stored ones
        self._derive_missing_raw()

        # fill in void regions in coarse levels (N.B., since ds_raw ignore q-selection, blocks generated for void areas will only contain selected q)
        self.restrict_data()

        # more post-processing
        self.mb_min = np.vstack([ds_raw['x1f'].min(axis=1), ds_raw['x2f'].min(axis=1), ds_raw['x3f'].min(axis=1) ]).T
        self.mb_max = np.vstack([ds_raw['x1f'].max(axis=1), ds_raw['x2f'].max(axis=1), ds_raw['x3f'].max(axis=1) ]).T

        self.lev_bbox = np.zeros([self.max_level+1, 6])  # bounding box for each level
        for _lev in range(0, self.max_level+1):
            _cecs = [unique_with_atol(np.sort(ds_raw['x1f'][self.levels == _lev].flatten())), 
                     unique_with_atol(np.sort(ds_raw['x2f'][self.levels == _lev].flatten())), 
                     unique_with_atol(np.sort(ds_raw['x3f'][self.levels == _lev].flatten()))]
            self.lev_bbox[_lev] = [_cecs[0].min(), _cecs[0].max(), _cecs[1].min(), _cecs[1].max(), _cecs[2].min(), _cecs[2].max()]

        # define abbreviated data name for easier access
        self._simplified_scalars = {
            # for hydro: cons have dens, mom1/2/3, prim have rho, vel1/2/3
            # for dust fluid: similarly species j have dust_j_[rho|dens], dust_j_[vel|mom]1/2/3
            "dens": ["dens", "rho", ],
            "rho": ["dens", "rho", ], 
            "rhog": ["dens", "rho", ], 
            "P": ["press", ],
            "pres": ["press", ],
            "pressure": ["press", ],
            "E": ["Etot", ]
        }
        self._density_choices = {"cons": "dens", "prim": "rho"}
        self._mv_choices = {"cons": "mom", "prim": "vel"}
        ## from Athena++'s code, it is possible to have output contains [vel|mom] or [vel|mom]_xyz as vectors
        self._tensor_header = {"u": "vel",  "v": "vel",  "M": "mom"}

        # determine geometric dimension symbols
        self._xyz_order = {'1': '1', '2': '2', '3': '3'}
        if ds_raw['Coordinates'] == b"cartesian":
            self.coor_name = "Car"
            self._xyz_order.update({'x': '1', 'y': '2', 'z': '3'})
        elif ds_raw['Coordinates'] == b"cylindrical":
            self.coor_name = "Cyl"
            self._xyz_order.update({'r': '1', 'phi': '2', 'z': '3', 'φ': '2', 'ϕ': '2'})
        elif ds_raw['Coordinates'] == b"spherical_polar":
            self.coor_name = "Sph"
            self._xyz_order.update({'r': '1', 'theta': '2', 'phi': '3', 'θ': '2', 'φ': '3', 'ϕ': '3'})
        else:
            warnings.warn(f"`__getitem__` with simplified component names (e.g., 'ur') is currently not supported under this coordinates: {ds_raw['Coordinates']}")
            
        # azimuthal dimensions that cover a full 2*pi, which get_level may wrap across
        self.periodic_span = self._periodic_dims()

        # finally, aggregate data at the lowest level by default     
        self.get_level(0)  # get_level now requires self.coor_name

    def _has_data(self, name):
        return name in self.ds or name in self.ds_raw

    def _get_direct_data(self, name):
        if name in self.ds:
            return self.ds[name].view()
        if name in self.ds_raw:
            return self.ds_raw[name]
        raise KeyError(name)

    def _warn_derivation(self, requested, expression):
        if self.warn_derived:
            warnings.warn(
                f"{requested} is not stored directly in {self.filename}; deriving as {expression}.",
                RuntimeWarning,
                stacklevel=3,
            )

    _RAW_DERIVABLE_RE = re.compile(
        r'(?:dust_(?P<species>\d+)_)?(?:(?P<field>vel|adv_vel|mom|diff_mom)(?P<comp>[1-3])|(?P<density>rho|dens))')

    def _derive_missing_raw(self):
        """ Add requested but unstored quantities to ds_raw, cell by cell, before restriction.

            A request like q=['rho', 'vel1', 'dust_1_vel1'] then works on a cons-only file or a
            combine_athdf_archive.py archive (cons + press + dust_N_diff_mom*), and q=['mom1'] on a prim
            file.  Velocities are momentum*(1/density), the way Athena forms its primitives (bitwise
            equal to the prim file for the same source precision); the dust velocity is the advective
            one, (dust_N_mom - dust_N_diff_mom)*(1/dust_N_dens).  Names that cannot be derived are left
            alone, so the usual KeyError still reports them.
        """
        raw = self.ds_raw

        def first(*names):
            return next((raw[n] for n in names if n in raw), None)

        def as_stored(value, like):
            return np.asarray(value).astype(np.asarray(like).dtype, copy=False)

        for q in self.quantities:
            if q in raw:
                continue
            match = self._RAW_DERIVABLE_RE.fullmatch(q)
            if match is None:
                continue
            species, field, comp = match.group("species"), match.group("field"), match.group("comp")
            pre = f"dust_{species}_" if species is not None else ""
            dens = first(pre + "dens", pre + "rho")
            if dens is None:
                continue
            if match.group("density") is not None:
                value, how = dens, f"{pre}dens/{pre}rho alias"
            elif species is None:                                      # gas
                mom, vel = first(f"mom{comp}"), first(f"vel{comp}")
                if field == "vel" and mom is not None:
                    value, how = as_stored(np.asarray(mom, np.float64)*(1.0/np.asarray(dens, np.float64)), mom), f"mom{comp}*(1/dens)"
                elif field == "mom" and vel is not None:
                    value, how = as_stored(np.asarray(dens, np.float64)*vel, vel), f"dens*vel{comp}"
                else:
                    continue
            else:                                                      # dust
                total, diff = first(f"{pre}mom{comp}"), first(f"{pre}diff_mom{comp}")
                adv = first(f"{pre}adv_vel{comp}", f"{pre}vel{comp}")
                d64 = np.asarray(dens, np.float64)
                if field in ("vel", "adv_vel") and adv is not None:
                    value, how = adv, f"{pre}adv_vel{comp}/{pre}vel{comp} alias"
                elif field in ("vel", "adv_vel") and total is not None and diff is not None:
                    value = as_stored((np.asarray(total, np.float64) - diff)*(1.0/d64), total)
                    how = f"({pre}mom{comp} - {pre}diff_mom{comp})*(1/{pre}dens)"
                elif field == "mom" and adv is not None and diff is not None:
                    value, how = as_stored(d64*adv + diff, adv), f"{pre}dens*{pre}vel{comp} + {pre}diff_mom{comp}"
                elif field == "diff_mom" and total is not None and adv is not None:
                    value, how = as_stored(np.asarray(total, np.float64) - d64*adv, total), f"{pre}mom{comp} - {pre}dens*{pre}vel{comp}"
                else:
                    continue
            raw[q] = value
            self._warn_derivation(q, how)

    def _safe_divide(self, numerator, denominator):
        try:
            return numerator / denominator
        except Exception:
            return numerator / denominator[..., None]

    def _component_index(self, comp):
        return self._xyz_order[comp]

    def _get_density(self, species):
        for real_wanted in (f"dust_{species}_dens", f"dust_{species}_rho"):
            if self._has_data(real_wanted):
                return self._get_direct_data(real_wanted)
        raise KeyError(f"Neither dust_{species}_dens nor dust_{species}_rho is available.")

    def _get_dust_adv_velocity_direct(self, species, comp_idx):
        """Return a directly stored advective velocity under either supported name."""
        for name in (f"dust_{species}_adv_vel{comp_idx}", f"dust_{species}_vel{comp_idx}"):
            if self._has_data(name):
                return name, self._get_direct_data(name)
        return None, None

    def _get_gas_component(self, requested):
        header = requested[0]
        comp = requested[1:]
        idx = self._component_index(comp)

        if header == "M":
            mom_wanted = "mom" + idx
            if self._has_data(mom_wanted):
                return self._get_direct_data(mom_wanted)
            vel_wanted = "vel" + idx
            for density_name in ("dens", "rho"):
                if self._has_data(vel_wanted) and self._has_data(density_name):
                    self._warn_derivation(requested, f"{density_name}*{vel_wanted}")
                    return self._get_direct_data(density_name) * self._get_direct_data(vel_wanted)
            raise KeyError(f"{requested} not found. {mom_wanted} not found, and cannot derive from dens*{vel_wanted}.")

        real_wanted = "vel" + idx
        if self._has_data(real_wanted):
            return self._get_direct_data(real_wanted)

        mom_wanted = "mom" + idx
        for density_name in ("dens", "rho"):
            if self._has_data(mom_wanted) and self._has_data(density_name):
                self._warn_derivation(requested, f"{mom_wanted}/{density_name}")
                return self._safe_divide(self._get_direct_data(mom_wanted), self._get_direct_data(density_name))
        raise KeyError(f"{requested} not found. {real_wanted} not found, and cannot derive from {mom_wanted}/dens.")

    def _parse_dust_component(self, name):
        match = re.match(
            r'^(?P<case>[wW])(?P<comp>[1-3]|[xyzr]|phi|theta|[φϕθ])(?:_(?P<kind>adv|diff|dif))?_(?P<species>\d+)$',
            name,
        )
        if match is None:
            raise KeyError(name)
        return match.group("case"), match.group("comp"), match.group("kind"), match.group("species")

    def _get_dust_momentum(self, requested, species, comp_idx, kind):
        total_name = f"dust_{species}_mom{comp_idx}"
        diff_name = f"dust_{species}_diff_mom{comp_idx}"
        adv_name = f"dust_{species}_adv_mom{comp_idx}"
        vel_name, adv_velocity = self._get_dust_adv_velocity_direct(species, comp_idx)

        if kind is None:
            if self._has_data(total_name):
                return self._get_direct_data(total_name)
            if self._has_data(adv_name) and self._has_data(diff_name):
                self._warn_derivation(requested, f"{adv_name} + {diff_name}")
                return self._get_direct_data(adv_name) + self._get_direct_data(diff_name)
            if vel_name is not None and self._has_data(diff_name):
                density = self._get_density(species)
                self._warn_derivation(requested, f"dust_{species}_dens_or_rho*{vel_name} + {diff_name}")
                return density * adv_velocity + self._get_direct_data(diff_name)
            raise KeyError(f"{requested} not found. {total_name} is unavailable and cannot be derived from advective plus diffusion momentum.")

        if kind in ("diff", "dif"):
            if self._has_data(diff_name):
                return self._get_direct_data(diff_name)
            if self._has_data(total_name) and self._has_data(adv_name):
                self._warn_derivation(requested, f"{total_name} - {adv_name}")
                return self._get_direct_data(total_name) - self._get_direct_data(adv_name)
            if self._has_data(total_name) and vel_name is not None:
                density = self._get_density(species)
                self._warn_derivation(requested, f"{total_name} - dust_{species}_dens_or_rho*{vel_name}")
                return self._get_direct_data(total_name) - density * adv_velocity
            raise KeyError(f"{requested} not found. {diff_name} is unavailable and cannot be derived.")

        if kind == "adv":
            if self._has_data(adv_name):
                return self._get_direct_data(adv_name)
            if vel_name is not None:
                density = self._get_density(species)
                self._warn_derivation(requested, f"dust_{species}_dens_or_rho*{vel_name}")
                return density * adv_velocity
            if self._has_data(total_name) and self._has_data(diff_name):
                self._warn_derivation(requested, f"{total_name} - {diff_name}")
                return self._get_direct_data(total_name) - self._get_direct_data(diff_name)
            raise KeyError(f"{requested} not found. {adv_name} is unavailable and cannot be derived.")

        raise KeyError(f"Unsupported dust momentum kind {kind!r} in {requested}.")

    def _get_dust_velocity(self, requested, species, comp_idx, kind):
        total_name = f"dust_{species}_mom{comp_idx}"
        diff_name = f"dust_{species}_diff_mom{comp_idx}"
        adv_name = f"dust_{species}_adv_mom{comp_idx}"
        vel_name, adv_velocity = self._get_dust_adv_velocity_direct(species, comp_idx)

        if kind == "adv":
            if vel_name is not None:
                return adv_velocity
            if self._has_data(adv_name):
                density = self._get_density(species)
                self._warn_derivation(requested, f"{adv_name}/dust_{species}_dens_or_rho")
                return self._safe_divide(self._get_direct_data(adv_name), density)
            if self._has_data(total_name) and self._has_data(diff_name):
                density = self._get_density(species)
                self._warn_derivation(requested, f"({total_name} - {diff_name})/dust_{species}_dens_or_rho")
                return self._safe_divide(self._get_direct_data(total_name) - self._get_direct_data(diff_name), density)
            raise KeyError(
                f"{requested} not found. Neither dust_{species}_adv_vel{comp_idx} nor dust_{species}_vel{comp_idx} "
                f"is stored, and it cannot be derived from {adv_name} or {total_name} - {diff_name}."
            )

        if kind in ("diff", "dif"):
            if self._has_data(diff_name):
                density = self._get_density(species)
                self._warn_derivation(requested, f"{diff_name}/dust_{species}_dens_or_rho")
                return self._safe_divide(self._get_direct_data(diff_name), density)
            if self._has_data(total_name) and self._has_data(adv_name):
                density = self._get_density(species)
                self._warn_derivation(requested, f"({total_name} - {adv_name})/dust_{species}_dens_or_rho")
                return self._safe_divide(self._get_direct_data(total_name) - self._get_direct_data(adv_name), density)
            if self._has_data(total_name) and vel_name is not None:
                density = self._get_density(species)
                self._warn_derivation(requested, f"{total_name}/dust_{species}_dens_or_rho - {vel_name}")
                return self._safe_divide(self._get_direct_data(total_name), density) - adv_velocity
            raise KeyError(f"{requested} not found. {diff_name} is unavailable and cannot be derived.")

        if self._has_data(total_name):
            density = self._get_density(species)
            self._warn_derivation(requested, f"{total_name}/dust_{species}_dens_or_rho")
            return self._safe_divide(self._get_direct_data(total_name), density)
        if self._has_data(adv_name) and self._has_data(diff_name):
            density = self._get_density(species)
            self._warn_derivation(requested, f"({adv_name} + {diff_name})/dust_{species}_dens_or_rho")
            return self._safe_divide(self._get_direct_data(adv_name) + self._get_direct_data(diff_name), density)
        if vel_name is not None and self._has_data(diff_name):
            density = self._get_density(species)
            self._warn_derivation(requested, f"{vel_name} + {diff_name}/dust_{species}_dens_or_rho")
            return adv_velocity + self._safe_divide(self._get_direct_data(diff_name), density)
        raise KeyError(f"{requested} not found. {total_name} is unavailable and total dust velocity cannot be derived from advective plus diffusion data.")

    def _get_named_dust_variable(self, requested):
        """Resolve a canonical dust name when its exact field is not stored."""
        match = re.fullmatch(
            r'dust_(?P<species>\d+)_(?:(?P<field>vel|mom|diff_mom|adv_mom|adv_vel)(?P<comp>[1-3])|(?P<density>rho|dens))',
            requested,
        )
        if match is None:
            raise KeyError(requested)
        species = match.group("species")
        if match.group("density") is not None:
            return self._get_density(species)

        field = match.group("field")
        comp_idx = match.group("comp")
        if field == "mom":
            return self._get_dust_momentum(requested, species, comp_idx, None)
        if field == "diff_mom":
            return self._get_dust_momentum(requested, species, comp_idx, "diff")
        if field == "adv_mom":
            return self._get_dust_momentum(requested, species, comp_idx, "adv")
        return self._get_dust_velocity(requested, species, comp_idx, "adv")

    def _get_dust_component(self, requested):
        case, comp, kind, species = self._parse_dust_component(requested)
        comp_idx = self._component_index(comp)
        if case == "W":
            return self._get_dust_momentum(requested, species, comp_idx, kind)
        return self._get_dust_velocity(requested, species, comp_idx, kind)

    def _get_magnitude(self, requested):
        if requested == "Mmag":
            return np.sqrt(np.sum(self["M"]**2, axis=-1))
        if requested in ("umag", "vmag"):
            return np.sqrt(np.sum(self[requested[0]]**2, axis=-1))

        match = re.match(r'^(?P<case>[Ww])mag(?:_(?P<kind>adv|diff|dif))?_(?P<species>\d+)$', requested)
        if match is None:
            raise KeyError(requested)
        case = match.group("case")
        kind = match.group("kind")
        species = match.group("species")
        total = None
        comps = ("1", "2") if self.ndim == 2 else ("1", "2", "3")
        for comp in comps:
            component_name = case + comp
            if kind is not None:
                component_name += "_" + kind
            component_name += "_" + species
            component = self[component_name]
            total = component * component if total is None else total + component * component
        return np.sqrt(total)
        
    def __getitem__(self, _name):
        """ Overload operator [] """
        
        if _name in self.ds:
            return self.ds[_name].view()  # ds will only contain np.ndarray
        elif _name in self.ds_raw:
            return self.ds_raw[_name]  # could be other type
        elif _name in self._simplified_scalars:
            for item in self._simplified_scalars[_name]:
                if item in self.ds:
                    return self.ds[item].view()
                elif item in self.ds_raw:
                    return self.ds_raw[item]
            raise KeyError(f"{_name} not found.", self._simplified_scalars[_name], "not found as well.")
        elif simplified_comp_check(_name):
            try:
                return self._get_gas_component(_name)
            except:
                raise KeyError(f"{_name} not found and cannot be derived.")
        elif _name in ['u', 'v', 'M',]:
            try:
                if self.ndim == 2:
                    return np.stack([self.__getitem__(_name+'1'), self.__getitem__(_name+'2')], axis=2)
                elif self.ndim == 3:
                    return np.stack([self.__getitem__(_name+'1'), self.__getitem__(_name+'2'), self.__getitem__(_name+'3')], axis=3)
                else:
                    raise RuntimeError("Unsupported ndim={self.ndim} for construct '{_name}'")
            except:
                raise KeyError(f"fail to construct full vector array: '{_name}'")
        elif magnitude_check(_name):
            try:
                return self._get_magnitude(_name)
            except:
                raise KeyError(f"fail to construct magnitude: '{_name}'")
        elif dust_var_check(_name):
            return self._get_named_dust_variable(_name)
        elif dust_simplified_check(_name):
            if _name[0] == "r":
                # currently we only accept rhod_j or rhop_j, where j is the species index
                return self._get_density(_name.split('_')[1])
            elif _name[0] == "w" or _name[0] == "W":
                # currently we only accept w1/x/r/..._j, where j is the species index
                try:
                    return self._get_dust_component(_name)
                except:
                    raise KeyError(f"{_name} not found and cannot be derived.")
        elif bool(re.match(r'^(w|W)_\d+$', _name)):  # for constructing 'w|W_number'
            try:
                _comp, _species = _name.split('_')
                if self.ndim == 2:
                    return np.stack([self.__getitem__(_comp+'1_'+_species), self.__getitem__(_comp+'2_'+_species)], axis=2)
                elif self.ndim == 3:
                    return np.stack([self.__getitem__(_comp+'1_'+_species), self.__getitem__(_comp+'2_'+_species), self.__getitem__(_comp+'3_'+_species)], axis=3)
                else:
                    raise RuntimeError("Unsupported ndim={self.ndim} for construct '{_name}'")
            except:
                raise KeyError(f"fail to construct full vector array: '{_name}'")
        else:
            raise KeyError(_name, " not found.")
    
    #def __setitem__(self, _name, _value):
    #    """ Overload assignment via operator [] """
    #    
    #    raise NotImplementedError("Writing access to data is not implemented.")
        
    def __contains__(self, _name):
        """ Overload operator `in` """
        
        if _name in self.ds:
            return True
        elif _name in self.ds_raw:
            return True
        else:
            return False

    def get_idx_mb(self, pos, lev=None):
        """ Get the MeshBlock index based on position """

        matched_idx_mbs = np.where(np.all((self.mb_min < pos) & (pos < self.mb_max), axis=1) == True)[0]
        if matched_idx_mbs.size == 0:  # all False
            raise ValueError("Seems the inquired position", pos, "is outside the domain.")
        if lev is None:
            idx_mb = matched_idx_mbs[np.argmax(self.levels[matched_idx_mbs])]
        else:
            if not np.any( self.levels[matched_idx_mbs] == lev ):
                print(f"Cannot find a MeshBlock on level {lev} for the inquired position", pos, "; using highest level")
                idx_mb = matched_idx_mbs[np.argmax(self.levels[matched_idx_mbs])]
            else:
                idx_mb = matched_idx_mbs[self.levels[matched_idx_mbs] == lev]
        return idx_mb

    def _periodic_dims(self):
        """ {dim: period} for the logical dimension that is azimuthal (x2 in Cyl, x3 in Sph) and spans 2*pi """
        dim = {"Cyl": 1, "Sph": 2}.get(getattr(self, "coor_name", None))
        if dim is None or dim >= self.ndim or self.root_grid_size[dim] <= 1:
            return {}
        x_min, x_max = self.ds_raw[f"RootGridX{dim+1}"][:2]
        if not np.isclose(x_max - x_min, 2 * np.pi, rtol=1e-6):
            return {}
        return {dim: float(x_max - x_min)}

    @staticmethod
    def _periodic_window(locs, n_blocks):
        """ (start, width), in blocks, of the smallest window holding every occupied index on a ring of n_blocks
            The plain [min, max] window is kept unless wrapping across the seam is strictly smaller, so a level
            that does not straddle the seam is laid out exactly as without wrapping.
        """
        occ = np.unique(locs)
        gaps = (np.roll(occ, -1) - occ - 1) % n_blocks  # empty blocks after each occupied one; gaps[-1] crosses the seam
        k = int(np.argmax(gaps))
        if gaps[k] <= gaps[-1]:
            return int(occ[0]), int(occ[-1] - occ[0] + 1)
        return int(occ[k + 1]), int(n_blocks - gaps[k])

    @staticmethod
    def _fill_coord_gaps(x, uniform):
        """ Fill NaN cell centres (slabs with no block on this level) from their neighbours:
            linear for uniformly spaced dimensions, log-linear otherwise.  Data there is empty anyway,
            so these values only position empty cells, but they keep the coordinate array monotonic.
        """
        bad = np.isnan(x)
        if not bad.any():
            return x
        idx = np.arange(x.size)
        if uniform or np.any(x[~bad] <= 0):
            x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        else:
            x[bad] = np.exp(np.interp(idx[bad], idx[~bad], np.log(x[~bad])))
        return x

    def wrap_to_level(self, x, dim):
        """ Map a periodic coordinate (e.g. a planet's phi in [0, 2*pi)) onto the working level's window along
            `dim`, whose coordinates may run past the upper domain edge (see get_level) """
        if dim not in self.periodic_span:
            return x
        lo = self._cecs[dim][0]
        return lo + np.mod(np.asarray(x) - lo, self.periodic_span[dim])

    def get_level(self, lev, roll_periodic=None):
        """ Construct the entire mesh data for a certain level
            CAVEAT: assuming mesh in that level is contiguous cuboid (either with rectilinear or curvilinear edges)
                    note that mesh may not be contiguous cuboid with AMR

            Aliases and derived quantities such as ur, wr_1, wr_adv_1, and
            wr_diff_1 are resolved from the current full-level data:
              1. If a requested quantity is directly stored in the current level
                 data, return that stored quantity.
              2. Otherwise, derive from the current level data. This naturally
                 prefers conservative density/momentum algebra when those fields
                 are available, e.g. ur=mom1/dens and wr_1=dust_N_mom1/dust_N_dens.
                 If total dust momentum is absent, it can also be reconstructed
                 from directly stored advective plus diffusion data.
              3. If only a physically unambiguous primitive field is available
                 for the request, use it, e.g. wr_adv_N from dust_N_vel* in a
                 pure prim file.
              4. If the requested physics is unavailable, raise KeyError instead
                 of silently substituting a different quantity.

            Periodic azimuthal dimension (phi spanning 2*pi, x2 in Cyl / x3 in Sph):
              The block window along phi is the smallest one holding all blocks of this level, and it wraps
              across the seam at the lower domain edge when that is smaller.  A refined region around phi = 0
              is then stored as one contiguous stretch instead of a full circle that is mostly empty (4x less
              memory for a two-planet run at level 2).  Coordinates past the seam are stored unwrapped (they
              run beyond the upper domain edge), so ccphi stays monotonic; use `wrap_to_level(phi, dim)` to map
              a phi into that range.  Levels that do not straddle the seam are laid out exactly as before.
              Pass roll_periodic=False here (or to the constructor) for the plain [min, max] block window.

            Cell-centre coordinates of slabs that hold no block of this level (possible inside the window)
            are filled from neighbouring slabs rather than left at 0; the data there stays masked/zero.
        """

        if roll_periodic is None:
            roll_periodic = self.kwargs.get("roll_periodic", True)

        sel_mb_lev = np.where(self.levels == lev)[0]
        logi_locs = self.logical_locations[sel_mb_lev].copy()
        Nx_mb = self.block_size
        wrapped = np.zeros(logi_locs.shape, dtype=bool)  # block lies past the seam of a periodic window
        for _d in range(3):
            if roll_periodic and _d in self.periodic_span:
                n_blocks = self.root_grid_size[_d] // Nx_mb[_d] * 2**lev
                anchor, _ = self._periodic_window(logi_locs[:, _d], n_blocks)
                wrapped[:, _d] = logi_locs[:, _d] < anchor
                logi_locs[:, _d] = (logi_locs[:, _d] - anchor) % n_blocks
            else:
                logi_locs[:, _d] -= logi_locs[:, _d].min()
        Nx_lev = Nx_mb * (logi_locs.max(axis=0) + 1)  # b/c locs starts from 0
        
        # reconstruct cell center coordinates (NaN marks slabs without a block on this level)
        ccx1, ccx2, ccx3 = np.full(Nx_lev[0], np.nan), np.full(Nx_lev[1], np.nan), np.full(Nx_lev[2], np.nan)
        #cex1, cex2, cex3 = np.zeros(Nx_lev[0]+1), np.zeros(Nx_lev[1]+1), np.zeros(Nx_lev[2]+1)
        
        if self.ndim == 2:
            Nx_lev = (Nx_lev[:2])
            
        level_data = {}
        for _q in self.quantities:
            level_data[_q] = np.zeros(Nx_lev[::-1], dtype=self.ds_raw[_q][0].dtype)
            if self.kwargs.get("AMRmask", False):
                level_data[_q] += np.nan  # a quick, dirty way to deal with irregular AMR level
        
        for idx_sel_mb, idx_mb in enumerate(sel_mb_lev):
            #print(idx_sel_mb, idx_mb)
            _ccx1, _ccx2, _ccx3 = self.ds_raw['x1v'][idx_mb], self.ds_raw['x2v'][idx_mb], self.ds_raw['x3v'][idx_mb]
            if wrapped[idx_sel_mb].any():  # unwrap blocks past the seam so the coordinates stay monotonic
                _ccx1, _ccx2, _ccx3 = [_c + self.periodic_span[_d] if wrapped[idx_sel_mb, _d] else _c
                                       for _d, _c in enumerate((_ccx1, _ccx2, _ccx3))]
            #_cex1, _cex2, _cex3 = self.ds_raw['x1f'][idx_mb], self.ds_raw['x2f'][idx_mb], self.ds_raw['x3f'][idx_mb]
            ccx1[Nx_mb[0]*logi_locs[idx_sel_mb][0]:Nx_mb[0]*(logi_locs[idx_sel_mb][0]+1)] = _ccx1
            ccx2[Nx_mb[1]*logi_locs[idx_sel_mb][1]:Nx_mb[1]*(logi_locs[idx_sel_mb][1]+1)] = _ccx2
            ccx3[Nx_mb[2]*logi_locs[idx_sel_mb][2]:Nx_mb[2]*(logi_locs[idx_sel_mb][2]+1)] = _ccx3
            
            for _q in self.quantities:
                if self.kwargs.get("same_level_mask", False) and idx_mb > self.old_num_mb:                    
                    _q_current_mb = np.nan
                else:
                    _q_current_mb = self.ds_raw[_q][idx_mb]
                if self.ndim == 2:
                    (level_data[_q])[Nx_mb[1]*logi_locs[idx_sel_mb][1]:Nx_mb[1]*(logi_locs[idx_sel_mb][1]+1),
                                     Nx_mb[0]*logi_locs[idx_sel_mb][0]:Nx_mb[0]*(logi_locs[idx_sel_mb][0]+1)] = _q_current_mb  #self.ds_raw[_q][idx_mb]
                        
                if self.ndim == 3:
                    (level_data[_q])[Nx_mb[2]*logi_locs[idx_sel_mb][2]:Nx_mb[2]*(logi_locs[idx_sel_mb][2]+1), 
                                     Nx_mb[1]*logi_locs[idx_sel_mb][1]:Nx_mb[1]*(logi_locs[idx_sel_mb][1]+1),
                                     Nx_mb[0]*logi_locs[idx_sel_mb][0]:Nx_mb[0]*(logi_locs[idx_sel_mb][0]+1)] = _q_current_mb  #self.ds_raw[_q][idx_mb]
                    
        for _q in self.quantities:
            level_data[_q] = np.ma.masked_invalid(level_data[_q])  # mask out missing data (will be transparent in colormap)
            if not level_data[_q].mask.any():    # Returns True if any of the elements of a evaluate to True (i.e., NaN exist)
                level_data[_q] = np.ma.getdata(level_data[_q])

        ccx1, ccx2, ccx3 = [self._fill_coord_gaps(_c, self.ds_raw[f"RootGridX{_d+1}"][2] == 1.0)
                            for _d, _c in enumerate((ccx1, ccx2, ccx3))]

        self._cecs = []
        for _d in range(3):
            _f = self.ds_raw[f'x{_d+1}f'][sel_mb_lev]
            if wrapped[:, _d].any():
                _f = _f + self.periodic_span[_d] * wrapped[:, _d][:, None]
            self._cecs.append(unique_with_atol(np.sort(_f.flatten())))
        
        # meaning: current "working level", its Nx, cell center coordinates, and level data set
        self.wlev, self.Nx_wlev, self.cccs, self.ds = lev, Nx_lev, [ccx1, ccx2, ccx3], level_data
        #return [ccx1, ccx2, ccx3], level_data  # uncomment this to turn on backward compatibility

        if self.coor_name == "Car":
            self.ccx, self.ccy, self.ccz = self.cccs
        elif self.coor_name == "Cyl":
            self.ccr, self.ccphi, self.ccz = self.cccs
        elif self.coor_name == "Sph":
            self.ccr, self.cctheta, self.ccphi = self.cccs
        else:
            pass
        
    _VELOCITY_RE = re.compile(r"(?:dust_(?P<species>\d+)_(?:adv_)?)?vel[1-3]")

    def _velocity_weight(self, q):
        """ Density that mass-weights velocity q under restriction; None if q is not a velocity, "" if no
            density is loaded to weight it with """
        match = self._VELOCITY_RE.fullmatch(q)
        if match is None:
            return None
        species = match.group("species")
        for name in (("rho", "dens") if species is None else (f"dust_{species}_rho", f"dust_{species}_dens")):
            if name in self.quantities:
                return name
        return ""

    def _mass_weighted_restriction(self, q, rho_name, idx_children, parent_loc, plain):
        """ sum(rho*v)/sum(rho) over the fine cells under each coarse cell of one restricted block
            (`plain`, the reader's plain mean, is kept where sum(rho) == 0) """
        shape = plain.shape  # (k, j, i)
        subsamples = [[slice(0, None, 2), slice(1, None, 2)] if n > 1 else [slice(None)] for n in shape]
        half = [n // 2 if n > 1 else n for n in shape]
        num, den = np.zeros(shape), np.zeros(shape)
        for idx in idx_children:
            offset = (self.logical_locations[idx] - 2 * np.asarray(parent_loc))[::-1]  # (k, j, i)
            dest = tuple(slice(o * h, (o + 1) * h) for o, h in zip(offset, half))
            rho = np.asarray(self.ds_raw[rho_name][idx], dtype=np.float64)
            mom = rho * np.asarray(self.ds_raw[q][idx], dtype=np.float64)
            for sk in subsamples[0]:
                for sj in subsamples[1]:
                    for si in subsamples[2]:
                        num[dest] += mom[sk, sj, si]
                        den[dest] += rho[sk, sj, si]
        out = np.array(plain, dtype=np.float64)
        np.divide(num, den, out=out, where=den > 0)
        return out.astype(plain.dtype)

    def restrict_data(self):
        """ Restrict data to complete void areas in coarse levels
            first, from the finest level to 2nd finest level, and then toward coarser levels

            Every field is restricted as the plain mean over the fine cells, except velocities (gas vel*,
            dust_N_vel*, dust_N_adv_vel*), which are mass-weighted: sum(rho*v)/sum(rho) with the matching
            gas or dust density. That is what Athena itself holds on a coarse cell (it restricts the
            conserved variables and converts), and it keeps a stored velocity equal to momentum/density
            on every level. Mass weighting applies to the default fast restriction; with subsample=True
            nothing is averaged, and with fast_restrict=False velocities stay volume-weighted means.
        """

        mass_weight = self.kwargs.get("fast_restrict", True) and not self.kwargs.get("subsample", False)
        velocity_weights = {_q: self._velocity_weight(_q) for _q in self.quantities}
        unweighted = [_q for _q, _rho in velocity_weights.items() if _rho == ""]
        if self.max_level > 0 and unweighted:
            warnings.warn(f"{unweighted} are restricted as plain means: no matching density is loaded to "
                          f"mass-weight them ({self.filename}).", RuntimeWarning, stacklevel=2)
        if self.max_level > 0 and not mass_weight and not self.kwargs.get("subsample", False) \
                and any(velocity_weights.values()):
            warnings.warn("fast_restrict=False: restricted velocities are volume-weighted means, not "
                          "mass-weighted.", RuntimeWarning, stacklevel=2)

        for lev in range(self.max_level, 0, -1):
            
            logi_locs = self.logical_locations[self.levels==lev]
            logi_locs_parent = logi_locs // 2  # get their logi_locs at -1 level

            # to find and group fine mesh blocks that can be merged into one coarse mesh blocks
            unq, count = np.unique(logi_locs_parent, axis=0, return_counts=True)
            repeated_groups = unq[count>1]

            re_levels = []
            re_logi_locs = []
            re_data = {'x1f': [], 'x1v': [], 'x2f': [], 'x2v': [], 'x3f': [], 'x3v': [], }
            for _q in self.quantities:
                re_data[_q] = []

            for repeated_group in repeated_groups:
                repeated_idx = np.argwhere(np.all(logi_locs_parent == repeated_group, axis=1))
                #print(repeated_idx.ravel()) # one can check this so we know it is 2D or 3D

                # hard-coded for 3D (but seems to also work in 2D so far)
                idx_to_merge = np.argwhere(self.levels==lev)[repeated_idx.ravel()].ravel()
                if idx_to_merge.size != 2**self.ndim:
                    print(f"Warning: len(idx_to_merge)={idx_to_merge.size} != 2^{self.ndim}."
                          f"\n         restrict at lev={lev} for meshblock {repeated_group} may fail."
                          f"\nFilename: {self.filename}")

                # athr.athdf uses face coordinates to find the enclosure boundaries, so center-coordiantes are fine to capture the mesh blocks
                bounding_box = np.array([[self.ds_raw['x1v'][idx_to_merge].min(), self.ds_raw['x1v'][idx_to_merge].max()], 
                                         [self.ds_raw['x2v'][idx_to_merge].min(), self.ds_raw['x2v'][idx_to_merge].max()], 
                                         [self.ds_raw['x3v'][idx_to_merge].min(), self.ds_raw['x3v'][idx_to_merge].max()], 
                                        ])
                #if lev == 5:
                #    np.set_printoptions(precision=17)
                #    print(repeated_idx.ravel(), repeated_group, idx_to_merge, '\n', bounding_box, bounding_box[:, 1]-bounding_box[:, 0])

                #if lev-1 < self.min_restrictable_level:  # identify levels that cannot be restricted by the finest level
                # or, we can speed up by only using +1 level to restrict new blocks
                if True:
                    update_pack = {'levels':self.levels, 'logical_locations':self.logical_locations, }
                    for _coord in ['x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', ]:
                        update_pack[_coord] = self.ds_raw[_coord]
                    for _q in self.quantities:
                        update_pack[_q] = self.ds_raw[_q] 
                else:
                    update_pack = None
                    
                # A mesh built by a user-defined MeshGenerator (input x?rat=-1, e.g. log-spaced radii) stores no
                # ratio in the file, so athr.athdf cannot rebuild its coordinates and raises "Must specify
                # user-defined face_func_?"; forward any face_func_?/center_func_? passed to Raw_Data_Restricted.
                # Meshes with x?rat>0 need none of these and are unaffected.
                coord_funcs = {_k: _v for _k, _v in self.kwargs.items()
                               if _k.startswith(("face_func_", "center_func_"))}
                _ds = athr.athdf(self.filename, level=lev-1,
                                 fast_restrict=self.kwargs.get("fast_restrict", True),
                                 subsample=self.kwargs.get("subsample", False),
                                 quantities=self.quantities,
                                 update_pack=update_pack,
                                 x1_min=bounding_box[0][0], x1_max=bounding_box[0][1],
                                 x2_min=bounding_box[1][0], x2_max=bounding_box[1][1],
                                 x3_min=bounding_box[2][0], x3_max=bounding_box[2][1], **coord_funcs)
                if mass_weight:
                    for _q, _rho in velocity_weights.items():
                        if _rho:
                            _ds[_q] = self._mass_weighted_restriction(_q, _rho, idx_to_merge, repeated_group, _ds[_q])

                re_levels.append(lev-1)
                re_logi_locs.append(repeated_group)
                for _coord in ['x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', ]:
                    re_data[_coord].append(_ds[_coord])
                for _q in self.quantities:
                    re_data[_q].append(_ds[_q])

            self.levels = np.hstack([self.levels, np.atleast_1d(re_levels)])
            self.logical_locations = np.vstack([self.logical_locations, np.array(re_logi_locs)])
            for _coord in ['x1f', 'x1v', 'x2f', 'x2v', 'x3f', 'x3v', ]:
                #try:
                self.ds_raw[_coord] = np.vstack([self.ds_raw[_coord], np.array(re_data[_coord])])
                #except Exception as e:
                #    print(lev, _coord, self.ds_raw[_coord].shape, re_data[_coord])
                #    raise ValueError(e)
                    
            for _q in self.quantities:
                self.ds_raw[_q] = np.vstack([self.ds_raw[_q], np.array(re_data[_q])])
                
        self.ds_raw['Levels'], self.ds_raw['LogicalLocations'] = self.levels, self.logical_locations
        self.ds_raw['NumMeshBlocks'] = self.logical_locations.shape[0]
        
    def get_ndim(self, num_ghost = 0):
        
        # mostly copied from athena_read.py
        nx_vals = []
        for d in range(3):
            if self.block_size[d] == 1 and self.root_grid_size[d] > 1:  # sum or slice
                other_locations = [location
                                   for location in zip(self.levels,
                                                       self.logical_locations[:, (d+1) % 3],
                                                       self.logical_locations[:, (d+2) % 3])]
                if len(set(other_locations)) == len(other_locations):  # effective slice
                    nx_vals.append(1)
                else:  # nontrivial sum
                    level = self.max_level
                    num_blocks_this_dim = 0
                    for level_this_dim, loc_this_dim in zip(self.levels,
                                                            self.logical_locations[:, d]):
                        if level_this_dim <= level:
                            possible_max = (loc_this_dim+1) * 2**(level-level_this_dim)
                            num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                        else:
                            possible_max = (loc_this_dim+1) // 2**(level_this_dim-level)
                            num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                    nx_vals.append(num_blocks_this_dim)
            elif self.block_size[d] == 1:  # singleton dimension
                nx_vals.append(1)
            else:  # normal case
                nx_vals.append(self.root_grid_size[d] * 2**self.max_level + 2 * num_ghost)
        nx1 = nx_vals[0]
        nx2 = nx_vals[1]
        nx3 = nx_vals[2]
        lx1 = nx1 // self.block_size[0]
        lx2 = nx2 // self.block_size[1]
        lx3 = nx3 // self.block_size[2]
        num_extended_dims = 0
        for nx in nx_vals:
            if nx > 1:
                num_extended_dims += 1
                
        return num_extended_dims
    
    def _plot_levels_old(self, ax=None, figsize=(10, 10), proj='z+'):

        if ax is None:
            new_ax_flag = True
            fig, ax = plt.subplots(figsize=figsize)
        
        if proj == 'z+':  # z+ means a view at top toward bottom, vertically
            for lev in range(0, self.max_level+1):
                ax.plot([self.lev_bbox[lev][0], self.lev_bbox[lev][1]], [self.lev_bbox[lev][2], self.lev_bbox[lev][2]], lw=2, alpha=0.65, c='C'+str(lev))
                ax.plot([self.lev_bbox[lev][0], self.lev_bbox[lev][1]], [self.lev_bbox[lev][3], self.lev_bbox[lev][3]], lw=2, alpha=0.65, c='C'+str(lev))
                ax.plot([self.lev_bbox[lev][0], self.lev_bbox[lev][0]], [self.lev_bbox[lev][2], self.lev_bbox[lev][3]], lw=2, alpha=0.65, c='C'+str(lev))
                ax.plot([self.lev_bbox[lev][1], self.lev_bbox[lev][1]], [self.lev_bbox[lev][2], self.lev_bbox[lev][3]], lw=2, alpha=0.65, c='C'+str(lev))
        else:
            raise NotImplementedError("Other perspective not yet implemented.")

        if new_ax_flag:
            fig.tight_layout()
            return fig, ax
    
    def plot_levels(self, ax=None, figsize=(12, 10), proj='z+'):
        """
        Plot bounding boxes of levels in different 2D projections.

        Assumes self.lev_bbox[lev] == [xmin, xmax, ymin, ymax, zmin, zmax].
        Projections:
        - 'z+': top view (x vs y)
        - 'x+': view along +x (y vs z)
        - 'y+': view along +y (x vs z)
        """
        import matplotlib.pyplot as plt

        new_ax_flag = ax is None
        if new_ax_flag:
            fig, ax = plt.subplots(figsize=figsize)

        # Indices for the min values of each plotted axis; max is +1 from each
        proj_axes = {
            'z+': (0, 2),  # x, y
            'x+': (2, 4),  # y, z
            'y+': (0, 4),  # x, z
        }
        axis_labels = {
            'z+': ('x', 'y'),
            'x+': ('y', 'z'),
            'y+': ('x', 'z'),
        }

        if proj not in proj_axes:
            raise NotImplementedError(f"Projection {proj} not implemented. Choose from {list(proj_axes.keys())}")

        i_a, i_b = proj_axes[proj]

        # Optional: track global bounds to make the plot nicely framed
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')

        for lev in range(0, self.max_level + 1):
            bbox = self.lev_bbox[lev]

            a0, a1 = bbox[i_a], bbox[i_a + 1]
            b0, b1 = bbox[i_b], bbox[i_b + 1]

            # draw rectangle for this level
            color = 'C' + str(lev % 10)
            ax.plot([a0, a1], [b0, b0], lw=2, alpha=0.65, c=color)
            ax.plot([a0, a1], [b1, b1], lw=2, alpha=0.65, c=color)
            ax.plot([a0, a0], [b0, b1], lw=2, alpha=0.65, c=color)
            ax.plot([a1, a1], [b0, b1], lw=2, alpha=0.65, c=color)

            xmin = min(xmin, a0); xmax = max(xmax, a1)
            ymin = min(ymin, b0); ymax = max(ymax, b1)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(axis_labels[proj][0])
        ax.set_ylabel(axis_labels[proj][1])
        if xmin < xmax and ymin < ymax:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        if new_ax_flag:
            fig.tight_layout()
            return fig, ax

        
    def _plot_mbs_old(self, ax=None, figsize=(10, 10), proj='z+'):

        if ax is None:
            new_ax_flag = True
            fig, ax = plt.subplots(figsize=figsize)
        
        if proj == 'z+':  # z+ means a view at top toward bottom, vertically
            for idx_mb in range(self.old_num_mb):
                _bbox = np.vstack([self.mb_min[idx_mb], self.mb_max[idx_mb]]).T.flatten()
                ax.plot([_bbox[0], _bbox[1]], [_bbox[2], _bbox[2]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
                ax.plot([_bbox[0], _bbox[1]], [_bbox[3], _bbox[3]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
                ax.plot([_bbox[0], _bbox[0]], [_bbox[2], _bbox[3]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
                ax.plot([_bbox[1], _bbox[1]], [_bbox[2], _bbox[3]], lw=2, alpha=0.4, c='C'+str(idx_mb % 10))
        else:
            raise NotImplementedError("Other perspective not yet implemented.")

        if new_ax_flag:
            fig.tight_layout()
            return fig, ax
        
    def plot_mbs(self, ax=None, figsize=(12, 10), proj='z+', sel_lev=None):
        """
        Plot bounding boxes of mesh blocks (MBs) in different 2D projections.

        Assumes:
        - self.mb_min[idx] = [xmin, ymin, zmin]
        - self.mb_max[idx] = [xmax, ymax, zmax]
        - self.old_num_mb is the number of MBs

        Projections:
        - 'z+': top view (x vs y)
        - 'x+': view along +x (y vs z)
        - 'y+': view along +y (x vs z)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        new_ax_flag = ax is None
        if new_ax_flag:
            fig, ax = plt.subplots(figsize=figsize)

        # Indices for the min values of each plotted axis; max is +1 from each
        proj_axes = {
            'z+': (0, 1),  # x, y
            'x+': (1, 2),  # y, z
            'y+': (0, 2),  # x, z
        }
        axis_labels = {
            'z+': ('x', 'y'),
            'x+': ('y', 'z'),
            'y+': ('x', 'z'),
        }

        if proj not in proj_axes:
            raise NotImplementedError(
                f"Projection {proj} not implemented. Choose from {list(proj_axes.keys())}"
            )

        i_a, i_b = proj_axes[proj]

        # Track global bounds for nice framing
        a_min = b_min = float('inf')
        a_max = b_max = float('-inf')

        # Draw each MB as a rectangle in the selected projection
        for mb_idx in range(self.old_num_mb):
            if sel_lev is not None:
                if self.levels[mb_idx] not in sel_lev:
                    continue

            # Build a 6-vector like [xmin, xmax, ymin, ymax, zmin, zmax]
            mbmin = np.asarray(self.mb_min[mb_idx], dtype=float)
            mbmax = np.asarray(self.mb_max[mb_idx], dtype=float)
            bbox = np.array([mbmin[0], mbmax[0], mbmin[1], mbmax[1], mbmin[2], mbmax[2]], dtype=float)

            a0, a1 = bbox[2*i_a], bbox[2*i_a + 1]
            b0, b1 = bbox[2*i_b], bbox[2*i_b + 1]

            color = 'C' + str(mb_idx % 10)
            ax.plot([a0, a1], [b0, b0], lw=2, alpha=0.4, c=color)
            ax.plot([a0, a1], [b1, b1], lw=2, alpha=0.4, c=color)
            ax.plot([a0, a0], [b0, b1], lw=2, alpha=0.4, c=color)
            ax.plot([a1, a1], [b0, b1], lw=2, alpha=0.4, c=color)

            a_min = min(a_min, a0); a_max = max(a_max, a1)
            b_min = min(b_min, b0); b_max = max(b_max, b1)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(axis_labels[proj][0])
        ax.set_ylabel(axis_labels[proj][1])
        if a_min < a_max and b_min < b_max:
            ax.set_xlim(a_min, a_max)
            ax.set_ylim(b_min, b_max)

        if new_ax_flag:
            fig.tight_layout()
            return fig, ax



def get_Q(ds, q):
    """
    Compute a derived quantity from `ds` based on a query string `q`.

    Grammar supported:
      - "<A>_over_<B>"           -> value(A) / value(B)
      - "<A>_over2_<B>"          -> value(A) / ds[B]   (special 'over2' form)
      - otherwise                -> ds[<token>]

    Simple tokens, including semantic aliases and magnitudes such as Mmag,
    umag, Wmag_1, and wmag_adv_1, are resolved by Raw_Data_Restricted.__getitem__.

    Broadcasting:
      - Tries A/B first; if it fails due to shape mismatch, retries with B[..., None].
    """

    def safe_divide(a, b):
        """Divide a by b; if shapes mismatch on the last axis, try b[..., None]."""
        try:
            return a / b
        except Exception:
            return a / b[..., None]

    def value_from_simple_token(tok):
        """Evaluate a token with no '_over' operators."""
        return ds[tok]

    def value_from_token(tok):
        """
        Evaluate a token that may include the special '_over2_' operator.
        '<A>_over2_<B>' means value(A) / value(B).
        """
        if '_over2_' in tok:
            a, b = tok.split('_over2_', 1)
            a_val = value_from_simple_token(a)
            b_val = value_from_simple_token(b)
            return safe_divide(a_val, b_val)
        # no over2: treat as simple
        return value_from_simple_token(tok)

    # Top-level: either a simple token or '<A>_over_<B>'
    if '_over_' in q:
        left, right = q.split('_over_', 1)
        q1 = value_from_token(left)
        q2 = value_from_token(right)
        return safe_divide(q1, q2)
    
    return value_from_simple_token(q)


def plot_slice(ds, lev, slice_coor='phi', slice_loc=np.pi, slice_idx=None, plot_method='pcolormesh',
               qleft='rho', qright='rhod_1', log10q=[True, True], 
               cmaps=['viridis', 'viridis'], vmins=[None, None], vmaxs=[None, None],
               plot_streamlines=False, sleft='M_over_rho', sright='W_1_over_rhod_1', add_OA_offset=False,
               circle_r=[0.08, 1.0], dist_factor=2, **kwargs):
    """
    Plot slices on selected quantities with optional streamlines.

    Parameters
    ----------
    ds : dataset object.
    lev : int -- refinement level.
    slice_coor : {'phi','theta','r'} -- which coordinate to slice.
    slice_loc : float -- slice location in the chosen coordinate (slice_idx can override)
    qleft, qright : str or array -- quantities to plot on left/right panel.
    log10q : [bool, bool] -- whether to log10-transform qleft/qright.
    cmaps, vmins, vmaxs : list -- colormap and normalization params for left/right.
    plot_streamlines : bool -- whether to overlay streamlines.
    sleft, sright : str -- quantities for streamlines.
    circle_r : list -- radii of circles to draw (scaled by R_H).
    dist_factor : float -- distance factor for selection around planet.
    plot_method : {'pcolormesh','imshow'} -- which plotting function to use.
    """
    
    rxplt.plt_params('paper'); fig, ax = plt.subplots(1, 2, figsize=(18, 8.25))
    ds.get_level(lev)
    planet_pos = kwargs.get("planet_pos", np.array([1.0, np.pi/2, np.pi]))
    R_H = kwargs.get("R_H", (1e-3/3)**(1/3))
    dist = dist_factor * R_H
    
    phi_p = ds.wrap_to_level(planet_pos[2], 2)  # phi of this level may run past 2*pi, see get_level
    sel_r     = (   ds.ccr  > planet_pos[0] - dist)&(    ds.ccr < planet_pos[0] + dist)
    sel_theta = (ds.cctheta > planet_pos[1] - dist)&(ds.cctheta < planet_pos[1] + dist)
    sel_phi   = (  ds.ccphi > phi_p - dist)&(  ds.ccphi < phi_p + dist)
    ccr, cctheta, ccphi = ds.ccr[sel_r], ds.cctheta[sel_theta], ds.ccphi[sel_phi]
    
    if slice_coor == 'phi':
        sel_phi = np.zeros(ds.ccphi.size, dtype=bool)
        if slice_idx is None:
            _phi_s = ds.wrap_to_level(slice_loc, 2)
            if not (ds._cecs[2][0] <= _phi_s <= ds._cecs[2][-1]):
                print(f"Warning: phi = {slice_loc} is outside level {lev}'s phi window "
                      f"[{ds._cecs[2][0]}, {ds._cecs[2][-1]}]; slicing at the nearest edge")
        id_sel_phi = slice_idx if slice_idx is not None else np.argmin(np.abs(ds.ccphi - _phi_s))
        sel_phi[id_sel_phi] = True; ccphi = ds.ccphi[sel_phi]
        print(f"Slicing on {slice_coor} at ds.ccphi[{id_sel_phi}]={ds.ccphi[sel_phi].item()}")
        x2plot, y2plot = ccr, cctheta; id_dim_on_slice = [0, 1]
        xlabel, ylabel = r"$r$", r"$\theta$"
        circle_pos = (planet_pos[0], planet_pos[1])
    elif slice_coor == "theta":
        sel_theta = np.zeros(ds.cctheta.size, dtype=bool)
        id_sel_theta = slice_idx if slice_idx is not None else np.argmin(np.abs(ds.cctheta - slice_loc))
        sel_theta[id_sel_theta] = True; cctheta = ds.cctheta[sel_theta]
        print(f"Slicing on {slice_coor} at ds.cctheta[{id_sel_theta}]={ds.cctheta[sel_theta].item()}")
        x2plot, y2plot = ccr, ccphi; id_dim_on_slice = [0, 2]
        xlabel, ylabel = r"$r$", r"$\phi$"
        circle_pos = (planet_pos[0], phi_p)
    elif slice_coor == "r":
        sel_r = np.zeros(ds.ccr.size, dtype=bool)
        id_sel_r = slice_idx if slice_idx is not None else np.argmin(np.abs(ds.ccr - slice_loc))
        sel_r[id_sel_r] = True; ccr = ds.ccr[sel_r]
        print(f"Slicing on {slice_coor} at ds.ccr[{id_sel_r}]={ds.ccr[sel_r].item()}")
        x2plot, y2plot = ccphi, cctheta; id_dim_on_slice = [2, 1]
        xlabel, ylabel = r"$\phi$", r"$\theta$"
        circle_pos = (phi_p, planet_pos[1])
    else:
        raise ValueError(f"Unknown slicing dimension: {slice_coor}")
    
    sel_slicing = np.ix_(sel_phi, sel_theta, sel_r)
    sel_slicing_vec = np.ix_(sel_phi, sel_theta, sel_r, [True]*ds.ndim)
    if isinstance(qleft, str):
        Qleft = (np.log10(get_Q(ds, qleft)) if log10q[0] else get_Q(ds, qleft))[sel_slicing].squeeze()
    else:
        Qleft = qleft[sel_slicing].squeeze(); Qleft = np.log10(Qleft) if log10q[0] else Qleft
    if isinstance(qright, str):
        Qright = (np.log10(get_Q(ds, qright)) if log10q[1] else get_Q(ds, qright))[sel_slicing].squeeze()
    else:
        Qright = qright[sel_slicing].squeeze(); Qright = np.log10(Qright) if log10q[1] else Qright
    if slice_coor == "r":
        Qleft = Qleft.T; Qright = Qright.T
        
    def plot_field(ax_, X, Y, Q, cmap, vmin, vmax, title, cax_title):
        if plot_method == "imshow":
            _half_dx, _half_dy = np.diff(X).mean()/2, np.diff(Y).mean()/2
            im = ax_.imshow(Q, origin="lower", extent=[X.min()-_half_dx, X.max()+_half_dx, Y.min()-_half_dy, Y.max()+_half_dy],
                            aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="None", rasterized=True,)
        else:  # default: pcolormesh
            im = ax_.pcolormesh(X, Y, Q, shading='auto', rasterized=True, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_.set(aspect=1.0, xlabel=xlabel, ylabel=ylabel, title=title)
        cax = rxplt.add_aligned_colorbar(fig, ax_, im, size="2%")
        cax.ax.set_title(cax_title, fontsize=24, y=1.025)

    # left and right panels
    plot_field(ax[0], x2plot, y2plot, Qleft,  cmaps[0], vmins[0], vmaxs[0], kwargs.get("title1", None), kwargs.get("cax_title1", None))
    plot_field(ax[1], x2plot, y2plot, Qright, cmaps[1], vmins[1], vmaxs[1], kwargs.get("title2", None), kwargs.get("cax_title2", None))

    """
    im = ax[0].pcolormesh(x2plot, y2plot, Qleft, shading='auto', rasterized=True, 
                          cmap=cmaps[0], vmin=vmins[0], vmax=vmaxs[0])
    ax[0].set(aspect=1.0, xlabel=xlabel, ylabel=ylabel, title=kwargs.get("title1", None), )
    cax = rxplt.add_aligned_colorbar(fig, ax[0], im, size="2%")
    cax.ax.set_title(kwargs.get("cax_title1", None), fontsize=24, y=1.025)
    
    im = ax[1].pcolormesh(x2plot, y2plot, Qright, shading='auto', rasterized=True, 
                          cmap=cmaps[1], vmin=vmins[1], vmax=vmaxs[1])
    ax[1].set(aspect=1.0, xlabel=xlabel, ylabel=ylabel, title=kwargs.get("title2", None), )
    cax = rxplt.add_aligned_colorbar(fig, ax[1], im, size="2%")
    cax.ax.set_title(kwargs.get("cax_title2", None), fontsize=24, y=1.025)
    """
    
    for _r in np.array(circle_r)*R_H:
        ax[0].add_patch(plt.Circle(circle_pos, _r, transform=ax[0].transData, fill=False, color='cyan', alpha=0.6, lw=2))
        ax[1].add_patch(plt.Circle(circle_pos, _r, transform=ax[1].transData, fill=False, color='cyan', alpha=0.6, lw=2))
        
    if plot_streamlines:
        # originally 3D arrays of vel vectors, transpose to (vec[r,theta,phi], sel_phi, sel_theta, sel_r)
        U = np.transpose(get_Q(ds, sleft)[sel_slicing_vec], (3, 0, 1, 2)).squeeze()
        V = np.transpose(get_Q(ds, sright)[sel_slicing_vec], (3, 0, 1, 2)).squeeze()
        if isinstance(add_OA_offset, bool):
            if add_OA_offset is True:  # default to 3D Model B
                h = (0.05*(ccr/1.0)**(-1/4) / np.sqrt(1/ccr**3)) / ccr
                OA_offset = (np.sqrt(1/ccr) * np.sqrt(1-11/4*h**2) - 0.993125)
                U[2] += OA_offset; V[2] += OA_offset
        else:
            try:
                U[2] += add_OA_offset; V[2] += add_OA_offset
            except Exception as e:
                print("Warning: cannot add OA_offset.\n", e)
        
        if slice_coor != 'r':
            _x2plot = np.linspace(ccr.min()+1e-10, ccr.max()-1e-10, ccr.size)  # uniform ccr
            U = np.transpose(U, (0, 2, 1)); V = np.transpose(V, (0, 2, 1));    # transpose to match x-y shape for Interp
        else:
            _x2plot = x2plot.copy()
        RGI_U1 = spint.RegularGridInterpolator((x2plot, y2plot), U[id_dim_on_slice[0]], method='linear', bounds_error=False)
        RGI_U2 = spint.RegularGridInterpolator((x2plot, y2plot), U[id_dim_on_slice[1]], method='linear', bounds_error=False)
        RGI_V1 = spint.RegularGridInterpolator((x2plot, y2plot), V[id_dim_on_slice[0]], method='linear', bounds_error=False)
        RGI_V2 = spint.RegularGridInterpolator((x2plot, y2plot), V[id_dim_on_slice[1]], method='linear', bounds_error=False)
        
        _ccx, _ccy = np.meshgrid(_x2plot, y2plot, indexing='ij')  # below transpose is needed for plotting purpose, RGI_XX(xsize, ysize).T
        ax[0].streamplot(_x2plot, y2plot, RGI_U1((_ccx, _ccy)).T, RGI_U2((_ccx, _ccy)).T, density=1.75, color=(1, 0.5, 0, 0.45))
        ax[1].streamplot(_x2plot, y2plot, RGI_V1((_ccx, _ccy)).T, RGI_V2((_ccx, _ccy)).T, density=1.75, color=(1, 0.5, 0, 0.45))
    
    if slice_coor == 'theta':
        ax[0].invert_xaxis(); ax[0].invert_yaxis(); ax[1].invert_xaxis(); ax[1].invert_yaxis()
    elif slice_coor == 'phi':
        ax[0].invert_xaxis(); ax[0].invert_yaxis(); ax[1].invert_xaxis(); ax[1].invert_yaxis()
    elif slice_coor == 'r':
        ax[0].invert_xaxis(); ax[1].invert_xaxis()
    
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":

    pass
