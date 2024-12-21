from PIL import Image
import random
from enum import Enum
import os
from collections import deque
import math
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime

# 地圖參數
WIDTH, HEIGHT = 46, 46  # 地圖大小 (單位：網格)
TILE_SIZE = 64  # 每張地形圖片大小

# 地形對應圖片
DATA_DIR = "data"
DEFAULT_MAP_FILE = f"{DATA_DIR}/default.map"
LOG_FILE = "log.txt"


class TerrainType(Enum):
    MOUNTAIN = "mountain.png"
    RIVER = "river.png"
    GRASS = "grass.png"
    ROCK = "rock.png"
    RIVERSTONE = "riverstone.png"

    def path(self):
        return f"{DATA_DIR}/{self.value}"


def read_map_from_file(file_path):
    type_dict = {
        "0": TerrainType.MOUNTAIN,
        "1": TerrainType.RIVER,
        "2": TerrainType.GRASS,
        "3": TerrainType.ROCK,
        "4": TerrainType.RIVERSTONE,
    }
    map_data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            row = [type_dict[ch] for ch in line.strip()]
            map_data.append(row)
    return map_data


# 隨機生成地圖
def generate_random_map():
    return [
        [random.choice(list(TerrainType)) for _ in range(WIDTH)] for _ in range(HEIGHT)
    ]


def connected_components_of_type(grid, target_types):
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右方向

    def bfs(start_row, start_col):
        """
        廣度優先搜索，找到從 (start_row, start_col) 開始的連通塊
        :return: 連通塊的大小和所有格子坐標
        """
        queue = deque([(start_row, start_col)])
        visited[start_row][start_col] = True
        component_size = 0
        component_cells = []

        while queue:
            r, c = queue.popleft()
            component_size += 1
            component_cells.append((r, c))

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    if grid[nr][nc] in target_types:  # 必須是目標地形
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        return component_size, component_cells

    component_pairs = []

    # 遍歷整個地圖
    for row in range(rows):
        for col in range(cols):
            if not visited[row][col] and grid[row][col] in target_types:
                component_size, component_cells = bfs(row, col)
                component_pairs.append((component_size, component_cells))

    return component_pairs


def gaussian_score(value, mu, sigma):
    # 高斯分布公式
    score = math.exp(-((value - mu) ** 2) / (2 * sigma**2))
    return score


def calculate_width(map_data, y, x, target_types):
    length_vertical = 1
    for dy, dx in [(-1, 0), (1, 0)]:
        ny, nx = y + dy, x + dx
        while 0 <= ny < HEIGHT and 0 <= nx < WIDTH and map_data[ny][nx] in target_types:
            length_vertical += 1
            ny += dy
            nx += dx

    length_horizontal = 1
    for dy, dx in [(0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        while 0 <= ny < HEIGHT and 0 <= nx < WIDTH and map_data[ny][nx] in target_types:
            length_horizontal += 1
            ny += dy
            nx += dx
    return min(length_vertical, length_horizontal)


def connect_score(map_data, target_types, sigma_factor):
    components = connected_components_of_type(map_data, target_types)
    sorted_components = sorted(components, key=lambda x: x[0], reverse=True)
    total_size = sum(size for size, _ in sorted_components)
    largest_size = sorted_components[0][0]

    return gaussian_score(largest_size, total_size, total_size / sigma_factor)


def width_score(map_data, target_types, who_to_punish, width_limit, sigma_factor):
    penalty = 0
    target_types_num = sum(
        1 for y in range(HEIGHT) for x in range(WIDTH) if map_data[y][x] in target_types
    )
    penalty_factor = 1 / target_types_num
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if map_data[y][x] in target_types:
                # use operator to determine the condition
                width = calculate_width(map_data, y, x, target_types)
                if who_to_punish == ">":
                    if width > width_limit:
                        penalty += penalty_factor * (
                            1
                            - gaussian_score(
                                width, width_limit, width_limit / sigma_factor
                            )
                        )
                elif who_to_punish == "<":
                    if width < width_limit:
                        penalty += penalty_factor * (
                            1
                            - gaussian_score(
                                width, width_limit, width_limit / sigma_factor
                            )
                        )
    return max(0, 1 - penalty)


def group_size_score(map_data, target_types, who_to_punish, group_limit, sigma_factor):
    components = connected_components_of_type(map_data, target_types)
    penalty = 0
    penalty_factor = 1 / len(components)
    for size, _ in components:
        if who_to_punish == ">":
            if size > group_limit:
                penalty += penalty_factor * (
                    1 - gaussian_score(size, group_limit, group_limit / sigma_factor)
                )
        elif who_to_punish == "<":
            if size < group_limit:
                penalty += penalty_factor * (
                    1 - gaussian_score(size, group_limit, group_limit / sigma_factor)
                )

    return max(0, 1 - penalty)


def group_number_score(map_data, target_types, group_limit, sigma):
    components = connected_components_of_type(map_data, target_types)
    return gaussian_score(len(components), group_limit, sigma)


def ratio_normalization(ratios):
    total = sum(ratios.values())
    return {terrain: count / total for terrain, count in ratios.items()}


def terrain_ratio(map_data):
    terrain_counts = {terrain: 0 for terrain in TerrainType}
    for row in map_data:
        for cell in row:
            terrain_counts[cell] += 1
    return ratio_normalization(terrain_counts)


def terrain_ratio_score(map_data, target_ratios, sigma):
    terrain_counts = terrain_ratio(map_data)
    total = sum(terrain_counts.values())
    score = 0
    for terrain, target_ratio in ratio_normalization(target_ratios).items():
        ratio = terrain_counts[terrain] / total
        score += gaussian_score(ratio, target_ratio, sigma) / len(target_ratios)

    return score


def valid_score(map_data, sigma_factor):
    valid_count = 0
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if is_valid_terrain(map_data, y, x, map_data[y][x]):
                valid_count += 1
    return gaussian_score(valid_count, HEIGHT * WIDTH, HEIGHT * WIDTH / sigma_factor)


# 包裝分數的處理函數
def handle_score(score, max_score, message):
    score = score * max_score
    print_str = f"{message} Score: {score}/{max_score}" + "\n"
    return score, print_str


# 適應度函數
def evaluate_fitness(map_data, print_details=False):
    print_str = ""
    print_str = "-" * 40 + "\n"

    max_score_dict = {
        "River Connect": 5e11,
        "Grass Connect": 1e12,
        "River Width": 1e5,
        "Grass Width": 1e3,
        "Mountain Width": 1e2,
        "Grass Group Size": 1e6,
        "Mountain Group Size": 1e7,
        "Rock Group Size": 1e1,
        "River Group Number": 1e4,
        "Grass Group Number": 1e8,
        "Mountain Group Number": 1e9,
        "Terrain Ratio": 1e13,
        "Valid": 1e10,
    }

    score_dict = {
        "River Connect": connect_score(
            map_data, [TerrainType.RIVER, TerrainType.RIVERSTONE], 1
        ),
        "Grass Connect": connect_score(
            map_data, [TerrainType.GRASS, TerrainType.RIVERSTONE], 1
        ),
        "River Width": width_score(
            map_data, [TerrainType.RIVER, TerrainType.RIVERSTONE], ">", 3, 2
        ),
        "Grass Width": width_score(
            map_data, [TerrainType.GRASS, TerrainType.ROCK], "<", 5, 2
        ),
        "Mountain Width": width_score(map_data, [TerrainType.MOUNTAIN], "<", 7, 1.5),
        "Grass Group Size": group_size_score(
            map_data, [TerrainType.GRASS], "<", 5 * 10, 1
        ),
        "Mountain Group Size": group_size_score(
            map_data, [TerrainType.MOUNTAIN], "<", 10 * 20, 1
        ),
        "Rock Group Size": group_size_score(map_data, [TerrainType.ROCK], ">", 3, 3),
        "River Group Number": group_number_score(
            map_data, [TerrainType.RIVER, TerrainType.RIVERSTONE], 1, 20
        ),
        "Grass Group Number": group_number_score(map_data, [TerrainType.GRASS], 7, 50),
        "Mountain Group Number": group_number_score(
            map_data, [TerrainType.MOUNTAIN], 5, 40
        ),
        "Terrain Ratio": terrain_ratio_score(
            map_data,
            {
                TerrainType.MOUNTAIN: 0.59,  # 1248
                TerrainType.RIVER: 0.1,  # 211
                TerrainType.GRASS: 0.29,  # 615
                TerrainType.ROCK: 0.01,  # 21
                TerrainType.RIVERSTONE: 0.001,  # 2
            },
            0.25,
        ),
        "Valid": valid_score(map_data, 10),
    }

    score = 0
    for key, value in score_dict.items():
        _score, _print_str = handle_score(value, max_score_dict[key], key)
        score += _score
        print_str += _print_str

    print_str += f"Total Score: {score}/{sum(max_score_dict.values())}" + "\n"
    print_str += "-" * 40 + "\n"
    if print_details:
        print(print_str)
    return score, print_str


def check_if(map_data, y, x, terrain_type):
    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in direction:
        ny, nx = y + dy, x + dx
        if 0 <= ny < HEIGHT and 0 <= nx < WIDTH and map_data[ny][nx] in terrain_type:
            return True
    return False


def is_valid_terrain(map_data, y, x, terrain):
    if terrain == TerrainType.RIVER:
        if check_if(map_data, y, x, [TerrainType.RIVER, TerrainType.RIVERSTONE]):
            return True
    elif terrain == TerrainType.RIVERSTONE:
        if check_if(map_data, y, x, [TerrainType.RIVER]) and check_if(
            map_data, y, x, [TerrainType.GRASS]
        ):
            return True
    elif terrain == TerrainType.GRASS:
        if check_if(map_data, y, x, [TerrainType.GRASS]):
            return True
    elif terrain == TerrainType.ROCK:
        if check_if(map_data, y, x, [TerrainType.GRASS]):
            return True
    elif terrain == TerrainType.MOUNTAIN:
        if check_if(map_data, y, x, [TerrainType.MOUNTAIN]):
            return True
    return False


def collect_valid_terrain(map_data, y, x):
    valid_terrain = []
    for terrain in TerrainType:
        if is_valid_terrain(map_data, y, x, terrain):
            valid_terrain.append(terrain)
    return valid_terrain


# 變異操作
def mutate(map_data, mutation_rate=0.01):
    tmp = deepcopy(map_data)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if random.random() < mutation_rate:
                if random.random() < 0.8:
                    choice = collect_valid_terrain(map_data, i, j)
                    for terrain in choice:
                        if (
                            terrain == TerrainType.RIVERSTONE
                            and random.random() < 0.9999
                        ):
                            choice.remove(terrain)
                        elif terrain == TerrainType.ROCK and random.random() < 0.99:
                            choice.remove(terrain)
                    if len(choice) > 0:
                        tmp[i][j] = random.choice(choice)
                else:
                    tmp[i][j] = random.choice(list(TerrainType))
    return tmp


def evolution_strategy(parent_map):
    for file in os.listdir(PROGRASS_DIR):
        os.remove(os.path.join(PROGRASS_DIR, file))

    best_overall_fitness = 0
    stagnation_counter = 0
    # print(evaluate_fitness(population[0], print_details=True))

    mutation_rate = MUTATION_RATE
    with tqdm(total=GENERATIONS) as pbar:
        for generation in range(GENERATIONS):
            population = [parent_map]
            while len(population) < POPULATION_SIZE:
                population.append(mutate(parent_map, mutation_rate))

            parent_map = sorted(
                population, key=lambda x: evaluate_fitness(x)[0], reverse=True
            )[0]

            # Calculate fitnesses
            best_fitness, print_str = evaluate_fitness(parent_map)

            if generation % 100 == 0:
                render_map_to_image(parent_map, f"{PROGRASS_DIR}/{generation}.png")
                with open(LOG_FILE, "a") as f:
                    f.write("Generation {:5d}\n".format(generation))
                    f.write(print_str)

            # Track best fitness
            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                stagnation_counter = 0
                mutation_rate = max(0.005, mutation_rate * 0.9)
            else:
                stagnation_counter += 1
                mutation_rate = min(0.1, mutation_rate * 1.1)

            pbar.set_description(
                "Generation {:5d} - best fitness: {:.2f}".format(
                    generation, best_overall_fitness
                )
            )
            pbar.update(1)

            if stagnation_counter > 50:
                break
    for i in range(4):
        render_map_to_image(parent_map, f"{PROGRASS_DIR}/{GENERATIONS + i}.png")
    create_gif_from_folder(
        PROGRASS_DIR, f"{GIF_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
    )
    return parent_map


# 將地圖合成為圖片
def render_map_to_image(map_data, output_file):
    # 加載地形圖片
    tile_images = {
        terrain_type: Image.open(terrain_type.path()) for terrain_type in TerrainType
    }

    # 創建空白地圖圖片
    map_image = Image.new("RGB", (WIDTH * TILE_SIZE, HEIGHT * TILE_SIZE))

    # 合成地圖
    for y, row in enumerate(map_data):
        for x, cell in enumerate(row):
            tile = tile_images[cell]
            map_image.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))

    # 保存結果圖片
    map_image.save(output_file)


def create_gif_from_folder(folder_path, output_path, duration=500, loop=0):
    """
    將資料夾中的圖片製作成 GIF
    :param folder_path: 資料夾路徑，包含圖片
    :param output_path: 輸出的 GIF 文件路徑
    :param duration: 每幀持續時間（毫秒）
    :param loop: 循環次數（0 表示無限循環）
    """
    # 獲取資料夾中的圖片文件，按名稱排序
    images = sorted(
        [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )

    if not images:
        print("資料夾中沒有找到圖片！")
        return

    # 打開圖片並將其添加到列表中
    frames = [Image.open(img) for img in images]

    # 創建 GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )


# 主程序
if __name__ == "__main__":
    OUTPUT_DIR = "output"
    PROGRASS_DIR = "prograss"
    GIF_DIR = "gif"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROGRASS_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)

    for file in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, file))

    POPULATION_SIZE = 20
    GENERATIONS = 10000
    MUTATION_RATE = 0.01

    # 生成地圖
    for i in range(10):
        parent_map = generate_random_map()
        # parent_map = read_map_from_file(DEFAULT_MAP_FILE)
        evaluate_fitness(parent_map, print_details=True)

        best_map = evolution_strategy(parent_map)
        evaluate_fitness(best_map, print_details=True)
        ratio = terrain_ratio(best_map)
        for terrain in TerrainType:
            component_size = [
                size for size, _ in connected_components_of_type(best_map, [terrain])
            ]
            print(f"{terrain.name}: {sorted(component_size, reverse=True)}")
            print(
                f"total: {int(ratio[terrain]*WIDTH*HEIGHT)}, len: {len(component_size)}"
            )

        render_map_to_image(best_map, f"{OUTPUT_DIR}/map_{i}.png")
