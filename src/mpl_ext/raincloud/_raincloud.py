#UTF-8 Encoding
"""
Raincloud plot. A custom matplotlib visual.

Used to investigate the distributions of features in data. For more
information, see:

https://wellcomeopenresearch.org/articles/4-63#:~:text=The%20raincloud
%20plot%20combines%20an,code%20to%20generate%20this%20figure.

Dependencies:
  - pandas
  - numpy
  - matplotlib

Created on: Tue 21 Mar 2023

@Author: HCoxJ
"""
### Imports
from typing import TypeVar, Union, Literal
from itertools import count
from pandas import DataFrame, Series
from matplotlib.axes import Axes
from numpy import where
from numpy.random import uniform

### Type Variables
_data_name = TypeVar('_data_name')
_group_name = TypeVar('_group_name')
_ord = TypeVar('_ord')

### Functions
def pyplot_cloud(
    df: DataFrame,
    col: _data_name,
    ax: Axes,
    group_by: Union[_group_name, None] = None,
    group_order: Union[callable, _group_name, dict[str, _ord], None] = None,
    reverse: bool = False,
    scale_clouds: Union[
        Literal['max'], dict[str, Union[int, float]], None
    ] = None,
    vert: bool = False,
    box_kwargs: Union[dict, None] = None,
    cloud_kwargs: Union[dict, None] = None
) -> dict[Literal['box', 'cloud', 'scats'], dict]:
    """
    Creates a cloud plot, a raincloud plot without the rain.

    Parameters
    ----------
    df: DataFrame.
        Contains the columns refenced in 'col' and 'group_by' parameters

    col: _data_name.
        Column name containing the data whose distribution is to be
        plotted. Data must be numeric.

    ax: Axes.
        Where the plot will be rendered.

    group_by: _group_name. Default: None.
        Column name containing categories to group the data by. The
        category names will be used as labels for each of the raincloud
        plots. Column must be string type.

    group_order: callable, _group_name, dictionary, or None.
    Default: None.
        Provides the order for each category to be plotted.

            _group_name: If group_by column name is passed, then the
            plots will be ordered by the category names.

            dictionary: mapping category names to some ordinal.

            callable: Some function that provides an ordinal output,
            used to order by some feature of the category's data. (e.g.
            passing len will order the categorys by their size)

    reverse: bool. Default: False.
        By default, the group ordering is ascending.

    scale_clouds: 'max' or dict[str, float] or None:
        Scales cloud sizes for each category.

            'max': clouds are scaled relative to the largest category's
            size.

            A dictionary of category names mapping to values between 0
            and 1: Each corresponding categories cloud will be scaled by
            the value given.

            None: clouds are not scaled.

    vert: bool. Default: False.
        Whether to render the plots vertically or horizontally.
        Overrides any 'vert' kwarg passed in box_kwargs, or
        violin_kwargs.

    box_kwargs: dict or None. Default: None.
        Additional arguments passed to boxplot. See matplotlib's boxplot
        documentation for details.

        NOTE: Any kwargs passed here will override this function's
        internal boxplot kwargs. In particular be careful with 'widths',
        as altering this value may cause visuals to overlap.

        Default kwargs:
            widths = 0.2,
            showfliers = False,
            showmeans = True,
            meanline = True,
            medianprops = dict(color = 'Black'),
            boxprops = dict(color = 'Black')

    cloud_kwargs: dict or None. Default: None.
        Additional arguments to modify the cloud componont. See
        matplotlib's violin plot documentation for details.

        NOTE: Any kwargs passed here will override this function's
        internal kwargs for vioilin plot. In particular be careful with
        'widths', as altering this value may cause visuals to overlap.

        Default kwargs:
            showmeans = False,
            showmedians = False,
            showextrema = False,
            widths = 1.2

    Returns
    -------
    dictionary containing the outputs of each visual for further
    customization.
    """
    if isinstance(group_by, type(None)):
        plot_data = [df[col].tolist()]
        label_names = [col]
    else:
        # extract groups according to group order
        if isinstance(group_order, type(None)):
            plot_data = {
                cat: df.loc[idx_vals, col].tolist()
                for cat, idx_vals
                in df.groupby(by = group_by, sort = False).groups.items()
            }
            label_names = list(plot_data.keys())
            plot_data = list(plot_data.values())

        elif isinstance(group_order, dict):
            plot_data: list[Series] = sorted(
                (
                    Series(df.loc[idx_vals, col], name = cat)
                    for cat, idx_vals
                    in df.groupby(by = group_by, sort = False).groups.items()
                ),
                key = lambda el: group_order[el.name],
                reverse = reverse
            )
            label_names = [el.name for el in plot_data]
            plot_data = [el.tolist() for el in plot_data]

        elif group_order == group_by:
            plot_data = {
                cat: df.loc[idx_vals, col].tolist()
                for cat, idx_vals
                in df.groupby(by = group_by, sort = False).groups.items()
            }
            label_names = sorted(plot_data.keys(), reverse = reverse)
            plot_data = list(plot_data[el] for el in label_names)

        elif callable(group_order):
            plot_data: list[Series] = sorted(
                (
                    Series(df.loc[idx_vals, col], name = cat)
                    for cat, idx_vals
                    in df.groupby(by = group_by, sort = False).groups.items()
                ),
                key = group_order,
                reverse = reverse
            )
            label_names = [el.name for el in plot_data]
            plot_data = [el.tolist() for el in plot_data]

        else:
            raise ValueError(
                f'group_order arg not recognised as valid: {group_order}'
            )

    # get cloud scalars
    if isinstance(scale_clouds, type(None)):
        scalars = [1 for _ in plot_data]

    elif scale_clouds == 'max':
        scalars = len(max(plot_data, key = len))
        scalars = [
            1 if len(el) == scalars else len(el)/scalars for el in plot_data
        ]

    elif isinstance(scale_clouds, dict):
        if len(
            err_vals:= {
                k: v for k, v in scale_clouds.items() if v < 0 or v > 1
            }
        ):
            raise ValueError(
                f'Some scalar values are out of bounds. {err_vals}'
            )

        scalars = [scale_clouds[el] for el in label_names]

    else:
        raise ValueError(
            f'scale_clouds arg not recognised as valid: {scale_clouds}'
        )

    # build boxplot
    if isinstance(box_kwargs, type(None)):
        bp = ax.boxplot(
            labels = label_names,
            x = plot_data,
            vert = vert,
            widths = 0.2,
            showfliers = False,
            showmeans = True,
            meanline = True,
            medianprops = dict(color = 'Black'),
            boxprops = dict(color = 'Black')
        )

    elif isinstance(box_kwargs, dict):
        # latter kwargs will take precedence over former:
        bp = {
            **dict(
                widths = 0.2,
                showfliers = False,
                showmeans = True,
                meanline = True,
                medianprops = dict(color = 'Black'),
                boxprops = dict(color = 'Black')
            ),
            **box_kwargs,
            **dict(
                labels = label_names,
                x = plot_data,
                vert = vert
            )
        }
        bp = ax.boxplot(**bp)

    else:
        raise ValueError(
            f'box_kwargs arg not recognised as valid: {box_kwargs}'
        )

    # build violin plot
    if isinstance(cloud_kwargs, type(None)):
        vp = ax.violinplot(
            dataset = plot_data,
            vert = vert,
            showmeans = False,
            showmedians = False,
            showextrema = False,
            widths = 1.2
        )

    elif isinstance(cloud_kwargs, dict):
        # latter kwargs will take precedence over former:
        vp = {
            **dict(
                showmeans = False,
                showmedians = False,
                showextrema = False,
                widths = 1.2
            ),
            **cloud_kwargs,
            **dict(
                dataset = plot_data,
                vert = vert
            )
        }
        vp = ax.violinplot(**vp)

    else:
        raise ValueError(
            f'cloud_kwargs arg not recognised as valid: {cloud_kwargs}'
        )
    
    for idx, scale, body in zip(count(), scalars, vp['bodies']):

        # Modify body to show only the upper half of the violin plot
        width_vals = body.get_paths()[0].vertices[:, 1]

        width_vals = scale * where(
            width_vals <= idx + 1,
            0,
            width_vals - idx - 1
        )

        # Update body
        body.get_paths()[0].vertices[:, 1] = width_vals + idx + 1

    return {'box': bp, 'cloud': vp}


def pyplot_raincloud(
    df: DataFrame,
    col: _data_name,
    ax: Axes,
    group_by: Union[_group_name, None] = None,
    group_order: Union[callable, _group_name, dict[str, _ord], None] = None,
    reverse: bool = False,
    scale_clouds: Union[
        Literal['max'], dict[str, Union[int, float]], None
    ] = None,
    vert: bool = False,
    box_kwargs: Union[dict, None] = None,
    cloud_kwargs: Union[dict, None] = None
) -> dict[Literal['box', 'cloud', 'scats'], dict]:
    """
    Creates a Raincloud plot.

    rain colour will always equal cloud colour.

    Parameters
    ----------
    df: DataFrame.
        Contains the columns refenced in 'col' and 'group_by' parameters

    col: _data_name.
        Column name containing the data whose distribution is to be
        plotted. Data must be numeric.

    ax: Axes.
        Where the plot will be rendered.

    group_by: _group_name. Default: None.
        Column name containing categories to group the data by. The
        category names will be used as labels for each of the raincloud
        plots. Column must be string type.

    group_order: callable, _group_name, dictionary, or None.
    Default: None.
        Provides the order for each category to be plotted.

            _group_name: If group_by column name is passed, then the
            plots will be ordered by the category names.

            dictionary: mapping category names to some ordinal.

            callable: Some function that provides an ordinal output,
            used to order by some feature of the category's data. (e.g.
            passing len will order the categorys by their size)

    reverse: bool. Default: False.
        By default, the group ordering is ascending.

    scale_clouds: 'max' or dict[str, float] or None:
        Scales cloud sizes for each category.

            'max': clouds are scaled relative to the largest category's
            size.

            A dictionary of category names mapping to values between 0
            and 1: Each corresponding categories cloud will be scaled by
            the value given.

            None: clouds are not scaled.

    vert: bool. Default: False.
        Whether to render the plots vertically or horizontally.
        Overrides any 'vert' kwarg passed in box_kwargs, or
        violin_kwargs.

    box_kwargs: dict or None. Default: None.
        Additional arguments passed to boxplot. See matplotlib's boxplot
        documentation for details.

        NOTE: Any kwargs passed here will override this function's
        internal boxplot kwargs. In particular be careful with 'widths',
        as altering this value may cause visuals to overlap.

        Default kwargs:
            widths = 0.2,
            showfliers = False,
            showmeans = True,
            meanline = True,
            medianprops = dict(color = 'Black'),
            boxprops = dict(color = 'Black')

    cloud_kwargs: dict or None. Default: None.
        Additional arguments to modify the cloud componont. See
        matplotlib's violin plot documentation for details.

        NOTE: Any kwargs passed here will override this function's
        internal kwargs for vioilin plot. In particular be careful with
        'widths', as altering this value may cause visuals to overlap.

        Default kwargs:
            showmeans = False,
            showmedians = False,
            showextrema = False,
            widths = 1.2

    Returns
    -------
    dictionary containing the outputs of each visual for further
    customization.
    """
    if isinstance(group_by, type(None)):
        plot_data = [df[col].tolist()]
        label_names = [col]
    else:
        # extract groups according to group order
        if isinstance(group_order, type(None)):
            plot_data = {
                cat: df.loc[idx_vals, col].tolist()
                for cat, idx_vals
                in df.groupby(by = group_by, sort = False).groups.items()
            }
            label_names = list(plot_data.keys())
            plot_data = list(plot_data.values())

        elif isinstance(group_order, dict):
            plot_data: list[Series] = sorted(
                (
                    Series(df.loc[idx_vals, col], name = cat)
                    for cat, idx_vals
                    in df.groupby(by = group_by, sort = False).groups.items()
                ),
                key = lambda el: group_order[el.name],
                reverse = reverse
            )
            label_names = [el.name for el in plot_data]
            plot_data = [el.tolist() for el in plot_data]

        elif group_order == group_by:
            plot_data = {
                cat: df.loc[idx_vals, col].tolist()
                for cat, idx_vals
                in df.groupby(by = group_by, sort = False).groups.items()
            }
            label_names = sorted(plot_data.keys(), reverse = reverse)
            plot_data = list(plot_data[el] for el in label_names)

        elif callable(group_order):
            plot_data: list[Series] = sorted(
                (
                    Series(df.loc[idx_vals, col], name = cat)
                    for cat, idx_vals
                    in df.groupby(by = group_by, sort = False).groups.items()
                ),
                key = group_order,
                reverse = reverse
            )
            label_names = [el.name for el in plot_data]
            plot_data = [el.tolist() for el in plot_data]

        else:
            raise ValueError(
                f'group_order arg not recognised as valid: {group_order}'
            )

    # get cloud scalars
    if isinstance(scale_clouds, type(None)):
        scalars = [1 for _ in plot_data]

    elif scale_clouds == 'max':
        scalars = len(max(plot_data, key = len))
        scalars = [
            1 if len(el) == scalars else len(el)/scalars for el in plot_data
        ]

    elif isinstance(scale_clouds, dict):
        if len(
            err_vals := {
                k: v for k, v in scale_clouds.items() if v < 0 or v > 1
            }
        ):
            raise ValueError(
                f'Some scalar values are out of bounds. {err_vals}'
            )

        scalars = [scale_clouds[el] for el in label_names]

    else:
        raise ValueError(
            f'scale_clouds arg not recognised as valid: {scale_clouds}'
        )

    # build boxplot
    if isinstance(box_kwargs, type(None)):
        bp = ax.boxplot(
            labels = label_names,
            x = plot_data,
            vert = vert,
            widths = 0.2,
            showfliers = False,
            showmeans = True,
            meanline = True,
            medianprops = dict(color = 'Black'),
            boxprops = dict(color = 'Black')
        )

    elif isinstance(box_kwargs, dict):
        # latter kwargs will take precedence over former:
        bp = {
            **dict(
                widths = 0.2,
                showfliers = False,
                showmeans = True,
                meanline = True,
                medianprops = dict(color = 'Black'),
                boxprops = dict(color = 'Black')
            ),
            **box_kwargs,
            **dict(
                labels = label_names,
                x = plot_data,
                vert = vert
            )
        }
        bp = ax.boxplot(**bp)

    else:
        raise ValueError(
            f'box_kwargs arg not recognised as valid: {box_kwargs}'
        )

    # build violin plot
    if isinstance(cloud_kwargs, type(None)):
        vp = ax.violinplot(
            dataset = plot_data,
            vert = vert
        )

    elif isinstance(cloud_kwargs, dict):
        # latter kwargs will take precedence over former:
        vp = {
            **dict(
                showmeans = False,
                showmedians = False,
                showextrema = False,
                widths = 1.2
            ),
            **cloud_kwargs,
            **dict(
                dataset = plot_data,
                vert = vert
            )
        }
        vp = ax.violinplot(**vp)

    else:
        raise ValueError(
            f'cloud_kwargs arg not recognised as valid: {cloud_kwargs}'
        )

    # modify violin plot and add scatters
    sc = dict()
    for idx, scale, body in zip(count(), scalars, vp['bodies']):

        # Modify body to show only the upper half of the violin plot
        width_vals = body.get_paths()[0].vertices[:, 1]

        width_vals = scale * where(
            width_vals <= idx + 1,
            0,
            width_vals - idx - 1
        )

        # Update body
        body.get_paths()[0].vertices[:, 1] = width_vals + idx + 1

        # get random y-values for jittered scatter plot
        jitter = uniform(
            low = -0.15,
            high = 0.15,
            size = len(data := plot_data[idx])
        )
        jitter += idx + 0.75

        # Add scatter plot
        sc[idx] = ax.scatter(
            x = data,
            y = jitter,
            s = .3,
            c = body.get_facecolor()
        )

    return {'box': bp, 'cloud': vp, 'scats': sc}