def location_to_index(location, spacing, shape):
    """Convert a location to an index.

    Parameters
    ----------
    location: tuple of float or None
        Location of source in m. If None is passed at a certain location
        of the tuple then the source is broadcast along the full extent
        of that axis. For example a source of `(0.1, 0.2, 0.1)` is a
        point source in 3D at the point x=10cm, y=20cm, z=10cm. A source of
        `(0.1, None, 0.1)` is a line source in 3D at x=10cm, z=10cm extending
        the full length of y.
    spacing : float
        Spacing of the grid in meters. The grid is assumed to be
        isotropic, all dimensions use the same spacing.
    shape : tuple of int
        Shape of the grid. This constrains the valid indices allowed.

    Returns
    -------
    tuple of int or slice
        Location of source in grid. Where source is broadcast along
        the whole axis a slice is used.
    """
    index = tuple(int(loc // spacing) if loc is not None else None for loc in location)

    # Ensure index is positive
    index = tuple(max(0, ind) if ind is not None else None for ind in index)

    # Ensure index is less than shape
    index = tuple(min(s-1, ind) if ind is not None else None for s, ind in zip(shape, index))

    index = tuple(ind if ind is not None else slice(None) for ind in index)

    return index