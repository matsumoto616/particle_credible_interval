import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

def draw_cumulative_distribution_fig(
    particles: np.ndarray,
    smoothe: bool = False
):
    """確率分布の粒子表現から、累積分布関数の形状を描画する

    Args:
        particles (np.ndarray): 粒子（確率分布からのiidサンプリング結果）
        smoothe (bool): Falseでは経験分布関数、Trueではそれを平滑化したものを描画する

    Returns:
        go.Figure: 描画結果
    """
    # 粒子数
    N = len(particles)

    # 範囲
    max_value = max(particles)
    min_value = min(particles)
    distribution_range = max_value - min_value

    # ソート
    sorted_particles = np.sort(particles)
    print(sorted_particles)

    # プロット
    fig = go.Figure()

    return fig

if __name__=="__main__":
    # テスト
    particles = [0, 11, 30, -5, 20, 10, 5]
    fig = draw_cumulative_distribution_fig(particles)


