"""Reader and writer for GOCAD TSurf triangulated surface files."""
import six
import numpy
import meshio

class tsurf(object):
    """
    Read, create, and write GOCAD TSurf triangulated surface files.

    Supports construction from a ``.ts`` file path or from raw coordinate
    and connectivity arrays.  Provides access to the triangulated mesh via
    the ``triangles`` property and can export back to the GOCAD TSurf format.

    Parameters
    ----------
    *args :
        Either a single ``filename`` string (reads from file), or four
        positional arguments ``x, y, z, cells`` (constructs from arrays).
    solid_color : tuple of float, optional
        RGBA colour tuple for visualisation.  Defaults to cyan ``(0,1,1,1)``.
    visible : str, optional
        GOCAD visibility flag string.  Defaults to ``"false"``.
    name : str, optional
        Surface name stored in the TSurf header.  Defaults to
        ``"Undefined"``.
    NAME : str, optional
        GOCAD coordinate system ``NAME`` field.  Defaults to
        ``"Default"``.
    AXIS_NAME : str, optional
        GOCAD coordinate system ``AXIS_NAME`` field.  Defaults to
        ``'"X" "Y" "Z"'``.
    AXIS_UNIT : str, optional
        GOCAD coordinate system ``AXIS_UNIT`` field.  Defaults to
        ``'"m" "m" "m"'``.
    ZPOSITIVE : str, optional
        GOCAD coordinate system ``ZPOSITIVE`` field.  Defaults to
        ``"Elevation"``.

    Attributes
    ----------
    mesh : meshio.Mesh
        Internal mesh representation holding ``points`` and ``cells``.
    x, y, z :
        Sequences of point coordinates.
    header : dict
        GOCAD header key-value pairs.
    csInfo : dict
        GOCAD coordinate system key-value pairs.
    name : str
        Surface name.
    solid_color : tuple
        RGBA visualisation colour.
    visible : str
        Visibility flag.

    Raises
    ------
    ValueError
        If the number of positional arguments is not 1 or 4.
    IOError
        If a filename is supplied that does not start with ``GOCAD TSurf``.
    """
    default_name = 'Undefined'
    default_solid_color = (0, 1, 1, 1.0)
    default_visible = 'false'
    default_NAME = 'Default'
    default_AXIS_NAME = '"X" "Y" "Z"'
    default_AXIS_UNIT = '"m" "m" "m"'
    default_ZPOSITIVE = 'Elevation'
    def __init__(self, *args, **kwargs):
        """
        Initialise a tsurf object from a file or from coordinate arrays.

        Accepts either a single filename or 4 arguments: x, y, z, cells.
        keyword arguments are: ``solid_color``, ``visible``, ``name``,
        ``NAME``, ``AXIS_NAME``, ``AXIS_UNIT``, ``ZPOSITIVE``.

        If a filename is given, the tsurf is read from the file.

        Otherwise, ``x``, ``y``, ``z`` are sequences of the x, y, and z
        coordinates of the points, and ``cells`` is a sequence of the
        indices of the coord arrays making up each triangle in the mesh,
        e.g. ``[[0, 1, 2], [2, 1, 3], ...]``.

        Parameters
        ----------
        *args :
            One string (filename) or four sequences (x, y, z, cells).
        **kwargs :
            Optional overrides for header and coordinate system fields.

        Raises
        ------
        ValueError
            If the number of positional arguments is not 1 or 4.
        """
        if len(args) == 1:
            self._read_tsurf(args[0])
        elif len(args) == 4:
            self._init_from_xyz(*args)
        else:
            raise ValueError('Invalid input arguments')
        color = kwargs.get('solid_color', None)
        visible = kwargs.get('visible', None)
        name = kwargs.get('name', None)
        if color is not None:
            self.solid_color = color
        if name is not None:
            self.name = name
        if visible is not None:
            self.visible = visible
        self.header['name'] = self.name
        self.header['solid*color'] = self.solid_color
        self.header['visible'] = self.visible

        NAME = kwargs.get('NAME')
        AXIS_NAME = kwargs.get('AXIS_NAME')
        AXIS_UNIT = kwargs.get('AXIS_UNIT')
        ZPOSITIVE = kwargs.get('ZPOSITIVE')

        if NAME is not None:
            self.NAME = NAME
        if AXIS_NAME is not None:
            self.AXIS_NAME = AXIS_NAME
        if AXIS_UNIT is not None:
            self.AXIS_UNIT = AXIS_UNIT
        if ZPOSITIVE is not None:
            self.ZPOSITIVE = ZPOSITIVE

        self.csInfo['NAME'] = self.NAME
        self.csInfo['AXIS_NAME'] = self.AXIS_NAME
        self.csInfo['AXIS_UNIT'] = self.AXIS_UNIT
        self.csInfo['ZPOSITIVE'] = self.ZPOSITIVE


    def _read_tsurf(self, filename):
        """
        Parse a GOCAD TSurf file and populate the instance attributes.

        Reads the HEADER block, GOCAD_ORIGINAL_COORDINATE_SYSTEM block,
        and the VRTX/PVRTX/TRGL data lines.

        Parameters
        ----------
        filename :
            Path to the GOCAD TSurf ``.ts`` file.

        Raises
        ------
        IOError
            If the file does not start with ``GOCAD TSurf``.
        """
        with open(filename, 'r') as infile:
            firstline = next(infile).strip()
            if not firstline.startswith('GOCAD TSurf'):
                raise IOError('This is not a valid TSurf file!')

            # Parse Header
            self.header = {}
            self.csInfo = {}
            line = next(infile).strip()
            if line.startswith('HEADER'):
                line = next(infile).strip()
                while '}' not in line:
                    key, value = line.split(':')
                    self.header[key.lstrip('*')] = value
                    line = next(infile).strip()
            self.name = self.header.get('name', filename)
            try:
                self.solid_color = [float(item) for item in self.header['solid*color'].split()]
                self.solid_color = tuple(self.solid_color)
            except KeyError:
                self.solid_color = self.default_solid_color
            try:
                self.visible = self.header['visible']
            except KeyError:
                self.visible = self.default_visible

            # Parse coordinate system info
            line = next(infile).strip()
            if line.startswith('GOCAD_ORIGINAL_COORDINATE_SYSTEM'):
                line = next(infile).strip()
                while 'END_ORIGINAL_COORDINATE_SYSTEM' not in line:
                    key, value = line.split(None, 1)
                    self.csInfo[key] = value
                    line = next(infile).strip()
            try:
                self.NAME = self.csInfo.get('NAME')
            except KeyError:
                self.NAME = self.default_NAME

            try:
                self.ZPOSITIVE = self.csInfo.get('ZPOSITIVE')
            except KeyError:
                self.ZPOSITIVE = self.default_ZPOSITIVE

            try:
                self.AXIS_NAME = self.csInfo.get('AXIS_NAME')
            except KeyError:
                self.AXIS_NAME = self.default_AXIS_NAME

            try:
                self.AXIS_UNIT = self.csInfo.get('AXIS_UNIT')
            except KeyError:
                self.AXIS_UNIT = self.default_AXIS_UNIT

            # Read points and cells
            # if not next(infile).startswith('TFACE'):
            #     raise IOError('Only "TFACE" format TSurf files are supported')
            points, cellArray = [], []
            for line in infile:
                line = line.strip().split()
                if line[0] in ['VRTX', 'PVRTX']:
                    points.append([float(item) for item in line[2:5]])
                elif line[0] == 'TRGL':
                    cellArray.append([int(item)-1 for item in line[1:]])
        self.x, self.y, self.z = zip(*points)
        points = numpy.array(points, dtype=numpy.float64)
        print(len(points))
        cellArray = numpy.array(cellArray, dtype=numpy.int)
        cells = [("triangle", cellArray)]
        self.mesh = meshio.Mesh(points, cells)

    @property
    def triangles(self):
        """
        Triangle vertex coordinates as a deduplicated array.

        Builds a lookup dictionary from vertex index to coordinate, then
        assembles each triangle's three corners into a row of 9 values
        ``[x1,y1,z1, x2,y2,z2, x3,y3,z3]``.

        Returns
        -------
        numpy.ndarray of shape (n_unique_triangles, 9)
            Unique triangles, sorted lexicographically by
            ``numpy.unique``.
        """
        triangle_numbers = self.mesh.cells[0].data
        vertex_dic = {i:vertex for i, vertex in enumerate(self.mesh.points)}
        triangle_array = numpy.array([numpy.hstack([vertex_dic[i] for i in triangle]) for triangle in triangle_numbers])
        return numpy.unique(triangle_array, axis=0)




    def _init_from_xyz(self, x, y, z, cells):
        """
        Initialise the tsurf from raw coordinate and cell arrays.

        Sets default header and coordinate system values and constructs
        the internal ``meshio.Mesh`` from the supplied data.

        Parameters
        ----------
        x :
            Sequence of x-coordinates of the mesh points.
        y :
            Sequence of y-coordinates of the mesh points.
        z :
            Sequence of z-coordinates of the mesh points.
        cells :
            Sequence of triangle connectivity lists, each containing
            three zero-based vertex indices, e.g.
            ``[[0, 1, 2], [2, 1, 3], ...]``.
        """
        points = numpy.array(list(zip(x, y, z)), dtype=numpy.float64)
        self.x, self.y, self.z = x, y, z
        cells = {"triangle": cells}
        self.solid_color = self.default_solid_color
        self.name = self.default_name
        self.visible = self.default_visible
        self.mesh = meshio.Mesh(points, cells)
        # Deleting the following default values since I have no idea
        # what they are.
        """
        self.header = {'moveAs':'2', 'drawAs':'2', 'line':'3',
                    'clip':'0', 'intersect':'0', 'intercolor':' 1 0 0 1'}
        """
        self.header = {}
        self.csInfo = {}
        self.NAME = self.default_NAME
        self.ZPOSITIVE = self.default_ZPOSITIVE
        self.AXIS_NAME = self.default_AXIS_NAME
        self.AXIS_UNIT = self.default_AXIS_UNIT

    def write(self, outname):
        """
        Write the tsurf to a GOCAD TSurf ``.ts`` file.

        Writes the HEADER block, GOCAD_ORIGINAL_COORDINATE_SYSTEM block,
        TFACE data (VRTX lines followed by TRGL lines), and the END
        marker.

        Parameters
        ----------
        outname :
            Output file path for the TSurf file.

        Notes
        -----
        Triangle connectivity is written with 1-based vertex indices as
        required by the GOCAD TSurf format specification.  Only the first
        cell block is written; multi-block meshes are not supported.
        """
        with open(outname, 'w') as outfile:
            # Write Header...
            outfile.write('GOCAD TSurf 1\n')
            outfile.write('HEADER {\n')
            """
            for key in ['name', 'color', 'moveAs', 'drawAs', 'line', 'clip',
                        'intersect', 'intercolor']:
                value = self.header[key]
            """
            for key, value in six.iteritems(self.header):
                if not isinstance(value, six.string_types):
                    try:
                        value = ' '.join(repr(item) for item in value)
                    except TypeError:
                        value = repr(value)
                if key == 'name':
                    outfile.write('{}:{}\n'.format(key, value))
                else:
                    outfile.write('*{}:{}\n'.format(key, value))
            outfile.write('}\n')

            # Write CS info...
            outfile.write('GOCAD_ORIGINAL_COORDINATE_SYSTEM\n')
            # for key, value in six.iteritems(self.csInfo):
            # It seems likely the CS keys should be in a particular order.
            for key in ['NAME', 'AXIS_NAME', 'AXIS_UNIT', 'ZPOSITIVE']:
                value = self.csInfo[key]
                if not isinstance(value, six.string_types):
                    try:
                        value = ' '.join(repr(item) for item in value)
                    except TypeError:
                        value = repr(value)
                outfile.write('{} {}\n'.format(key, value))
            outfile.write('END_ORIGINAL_COORDINATE_SYSTEM\n')

            # Write data...
            outfile.write('TFACE\n')
            for i, (x, y, z) in enumerate(self.mesh.points, start=1):
                template = '\t'.join(['VRTX {}'] + 3*['{: >9.3f}']) + '\n'
                outfile.write(template.format(i, x, y, z))
            # For now, assume only one set of cells, and that they are all
            # triangles.
            for a, b, c in self.mesh.cells[0].data:
                outfile.write('TRGL {} {} {}\n'.format(a+1, b+1, c+1))
            outfile.write('END\n')
