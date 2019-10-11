import torch


def create_vis_plot(vis, X_, Y_, title_, legend_):
    return vis.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, len(legend_))).cpu(),
        opts=dict(
            xlabel=X_,
            ylabel=Y_,
            title=title_,
            legend=legend_
        )
    )


def update_vis_plot(vis, item, loss, window, update_type):
    if item == 0:
        update_type = True

    vis.line(
        X=torch.ones((1, len(loss))).cpu() * item,
        Y=torch.Tensor(loss).unsqueeze(0).cpu(),
        win=window,
        update=update_type
    )
