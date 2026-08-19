# -*- coding: utf-8 -*-
"""

Read and write tab-separated tables and matrix-shaped outputs.

Functions
---------
write_tsv
    Write a table or labeled matrix to a TSV file.
read_tsv
    Read a TSV file as a table or reconstruct its labeled matrix structure.

Federico Ramírez-Toraño
09/06/2022

"""

import pandas


def write_tsv(table, filename,float_format=None):
    """
    Write a table or labeled matrix to a TSV file.

    Parameters
    ----------
    table : pandas.DataFrame | dict
        Table or labeled matrix to write.
    filename : str | pathlib.Path
        Path of the TSV file.
    float_format : str | None
        Format used to write floating-point values.
    """

    # If the input is a dictionary, converts it into a Pandas data frame.
    if isinstance(table, dict):
        table = pandas.DataFrame(table['matrix'], index=table['rows'], columns=table['columns'])

        # Marks the data as a matrix.
        ismatrix = True

    else:
        ismatrix = False

    # Checks that the input is a Pandas data frame.
    assert isinstance(table, pandas.DataFrame), 'Invalid input data.'

    # Writes the data as a TSV file.
    table.to_csv(
        filename,
        sep='\t',
        na_rep='n/a',
        index=ismatrix,
        index_label='output_name',
        encoding='utf-8-sig',
        float_format=float_format
    )


# Helper function to read TSV data.
def read_tsv(filename, ismatrix=False):
    """
    Read a TSV file as a table or reconstruct its labeled matrix structure.

    Parameters
    ----------
    filename : str | pathlib.Path
        Path of the TSV file.
    ismatrix : bool
        Whether to reconstruct a labeled matrix representation.

    Returns
    -------
    table : pandas.DataFrame | dict
        Loaded table or labeled matrix.
    """

    # If the TSV contains a matrix the first column is the row label.
    if ismatrix:
        index_col = 0
    else:
        index_col = None

    # Reads the table as a DataFrame object.
    table = pandas.read_table(filename, delimiter='\t', index_col=index_col, comment=None, encoding='utf-8-sig')

    # If the TSV contains a matrix rewrites the output.
    if ismatrix:
        table = {'rows': table.index, 'columns': table.columns, 'matrix': table.values}

    # Returns the read table.
    return table
