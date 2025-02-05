// 引入依赖：在 Cargo.toml 中添加 lazy_static = "1.4" 和 num_cpus = "1.13"
#[macro_use]
extern crate lazy_static;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// 定义局面状态
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalcChildState {
    Waiting,    // 等待子节点计算
    Processing, // 正在计算
    Finished,   // 计算完成
}

// 为了用于 HashMap 去重，需要为 Buju 实现 Hash 和 Eq
// 此处仅提供简单示例，实际需要根据局面状态（主要是 blocks 和 grid）确定唯一性
use std::hash::{Hash, Hasher};
#[derive(Clone, Debug)]
pub struct Buju {
    // 所有滑块信息：(类型, 横坐标, 纵坐标)
    pub blocks: Vec<(u32, u32, u32)>,
    // 棋盘信息，0表示空白，非0数字代表滑块的类型
    pub grid: [[u16; 6]; 6],
    // 父局面在全局列表中的索引
    pub father: Option<usize>,
    // 当前局面的状态
    pub state: CalcChildState,
}

impl PartialEq for Buju {
    fn eq(&self, other: &Self) -> bool {
        // 此处只以 grid 作为判重依据，也可以结合 blocks 信息
        self.grid == other.grid
    }
}

impl Eq for Buju {}

impl Hash for Buju {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // 仅使用 grid 进行 hash
        for row in &self.grid {
            row.hash(state);
        }
    }
}

// 全局数据存储
lazy_static! {
    // 所有局面存放在此处：初始局面放在下标 0，子局面逐步加入
    pub static ref BUJUS: Mutex<Vec<Buju>> = Mutex::new(Vec::new());
    // 去重 HashMap：key 为局面，value 为局面在 BUJUS 中的索引
    pub static ref BUJUMAP: Mutex<HashMap<Buju, usize>> = Mutex::new(HashMap::new());
}

// 判断当前局面是否为解题状态
fn is_solution(buju: &Buju) -> bool {
    // 示例：假设目标滑块为 blocks 中类型为 1 的滑块，
    // 当其右侧与出口对齐时，则认为解题成功。
    // 实际判断需要根据具体游戏规则，比如目标滑块是否移动到右侧第3行的出口位置。
    // 此处仅作示例判断：
    for &(typ, x, y) in &buju.blocks {
        if typ == 1 {
            // 假设 x 表示横坐标（0起始），当 x + 2 == 6 时说明到达右边界，
            // 并且 y == 2（即第三行，0起始）时则算解题
            if x + 2 == 6 && y == 2 {
                return true;
            }
        }
    }
    false
}

// 根据当前局面生成所有可能的子局面
fn generate_children(index: usize, current: &Buju) -> Vec<Buju> {
    let mut children = Vec::new();

    // 这里遍历所有滑块，根据滑块类型（横向或纵向）、当前位置，
    // 尝试往所有合法方向移动一个单位，并判断移动后是否与其他块重叠。
    // 下面仅给出伪代码，实际需要实现具体规则：
    for (block_idx, &(typ, x, y)) in current.blocks.iter().enumerate() {
        // 根据类型区分横向（假设类型 1 为目标及其他横向滑块）和纵向
        if typ == 1 {
            // 横向滑块只能左右移动
            // 试着左移
            if x > 0 && current.grid[y as usize][(x - 1) as usize] == 0 {
                // 创建新的局面，更新 blocks 和 grid
                let mut new_blocks = current.blocks.clone();
                new_blocks[block_idx].1 = x - 1;
                let new_grid = update_grid(&new_blocks);
                let child = Buju {
                    blocks: new_blocks,
                    grid: new_grid,
                    father: Some(index),
                    state: CalcChildState::Waiting,
                };
                children.push(child);
            }
            // 试着右移
            if x + 2 < 6 && current.grid[y as usize][(x + 2) as usize] == 0 {
                let mut new_blocks = current.blocks.clone();
                new_blocks[block_idx].1 = x + 1;
                let new_grid = update_grid(&new_blocks);
                let child = Buju {
                    blocks: new_blocks,
                    grid: new_grid,
                    father: Some(index),
                    state: CalcChildState::Waiting,
                };
                children.push(child);
            }
        } else {
            // 假设其他类型均为纵向（或其他横向）
            // 根据类型判断移动方向，此处只给出伪代码
            // 例如，若是纵向，则只能上下移动
            // 上移
            if y > 0 && current.grid[(y - 1) as usize][x as usize] == 0 {
                let mut new_blocks = current.blocks.clone();
                new_blocks[block_idx].2 = y - 1;
                let new_grid = update_grid(&new_blocks);
                let child = Buju {
                    blocks: new_blocks,
                    grid: new_grid,
                    father: Some(index),
                    state: CalcChildState::Waiting,
                };
                children.push(child);
            }
            // 下移（根据块高度判断是否越界）
            // 这里假设每个块高度为 1 或更多，需要根据当前块的实际高度判断末尾位置
            // 简化示例：下移一个单位
            if y + 1 < 6 && current.grid[(y + 1) as usize][x as usize] == 0 {
                let mut new_blocks = current.blocks.clone();
                new_blocks[block_idx].2 = y + 1;
                let new_grid = update_grid(&new_blocks);
                let child = Buju {
                    blocks: new_blocks,
                    grid: new_grid,
                    father: Some(index),
                    state: CalcChildState::Waiting,
                };
                children.push(child);
            }
        }
    }

    children
}

/// 根据滑块的信息返回该滑块的尺寸 (宽, 高)
fn block_size(block: &(u32, u32, u32)) -> (u32, u32) {
    let (typ, _x, _y) = *block;
    match typ {
        1 => (2, 1),
        2 => (2, 1),
        3 => (1, 2),
        4 => (3, 1),
        5 => (1, 3),
        _ => (0, 0),
    }
}

/// 根据 blocks 信息重建棋盘 grid，
/// 将每个滑块在棋盘上占用的格子标记为该滑块的类型，
/// 其余位置置为 0
fn update_grid(blocks: &[(u32, u32, u32)]) -> [[u16; 6]; 6] {
    // 初始化全部置 0 的棋盘
    let mut grid = [[0u16; 6]; 6];
    let mut count = 0u16;
    // 遍历每个滑块，根据其起始位置和尺寸，标记其占据的格子
    for &(typ, x, y) in blocks.iter() {
        let (w, h) = block_size(&(typ, x, y));
        let tid = typ as u16 * 100 + count;
        count += 1;
        // 对于横向滑块，从 (x, y) 开始，横向延伸 w 个格子，纵向延伸 h 个格子
        for j in 0..h {
            for i in 0..w {
                let grid_x = (x + i) as usize;
                let grid_y = (y + j) as usize;
                if grid_x < 6 && grid_y < 6 {
                    grid[grid_y][grid_x] = tid;
                } else {
                    // 若出现越界情况，说明该局面本身就不合法，
                    // 这里可以根据需要进行处理（例如打印警告或者直接 panic）
                    eprintln!(
                        "警告：滑块类型 {} 放置位置 ({},{}) 尺寸 ({},{}) 越界",
                        typ, x, y, w, h
                    );
                }
            }
        }
    }
    println!("=================");
    for g in &grid {
        println!("{:03?}", g);
    }
    println!("=================");
    grid
}

fn main() {
    // 初始化初始局面
    let initial_blocks = vec![
        // 例如：目标滑块类型设为1，位于 (1,2)（0起始），占据 (1,2) 和 (2,2)
        (1, 1, 2),
        // 其它滑块：类型2、3… 根据需要添加
        (2, 0, 0),
        (3, 3, 4),
    ];
    let initial_grid = update_grid(&initial_blocks);
    let initial_buju = Buju {
        blocks: initial_blocks,
        grid: initial_grid,
        father: None,
        state: CalcChildState::Waiting,
    };

    {
        let mut bujus = BUJUS.lock().unwrap();
        bujus.push(initial_buju.clone());
        let mut bujumap = BUJUMAP.lock().unwrap();
        bujumap.insert(initial_buju, 0);
    }

    // 使用多线程进行局面扩展，线程数量根据 CPU 核心数设置
    let num_threads = num_cpus::get();
    let mut handles = Vec::new();

    // 使用一个原子变量用于标记是否已找到解
    let solution_found = Arc::new(Mutex::new(None)); // 存放解题局面在 BUJUS 中的索引

    for _ in 0..num_threads {
        let solution_found = Arc::clone(&solution_found);
        let handle = thread::spawn(move || {
            loop {
                // 先判断是否已有解
                {
                    let sol = solution_found.lock().unwrap();
                    if sol.is_some() {
                        break;
                    }
                }
                let current_index_opt = {
                    // 从全局 BUJUS 中查找一个状态为 Waiting 的局面，并将其标记为 Processing
                    let mut bujus = BUJUS.lock().unwrap();
                    let mut index_opt = None;
                    for (i, buju) in bujus.iter_mut().enumerate() {
                        if buju.state == CalcChildState::Waiting {
                            buju.state = CalcChildState::Processing;
                            index_opt = Some(i);
                            break;
                        }
                    }
                    index_opt
                };

                // 如果没有找到等待处理的局面，则短暂等待再继续
                if let Some(index) = current_index_opt {
                    // 获取当前局面
                    let current = {
                        let bujus = BUJUS.lock().unwrap();
                        bujus[index].clone()
                    };

                    // 如果当前局面已经是解，则记录，并退出线程
                    if is_solution(&current) {
                        let mut sol = solution_found.lock().unwrap();
                        *sol = Some(index);
                        break;
                    }

                    // 生成子局面
                    let children = generate_children(index, &current);

                    // 对每个子局面进行去重判断后加入全局数据
                    for child in children {
                        // 锁定全局 BUJUMAP 进行检查
                        let mut add_new = false;
                        {
                            let mut map = BUJUMAP.lock().unwrap();
                            if !map.contains_key(&child) {
                                add_new = true;
                            }
                        }
                        if add_new {
                            // 加入全局 BUJUS 和 BUJUMAP
                            let mut buju_index = 0;
                            {
                                let mut bujus = BUJUS.lock().unwrap();
                                buju_index = bujus.len();
                                bujus.push(child.clone());
                            }
                            let mut map = BUJUMAP.lock().unwrap();
                            map.insert(child, buju_index);
                        }
                    }

                    // 标记当前局面为 Finished
                    {
                        let mut bujus = BUJUS.lock().unwrap();
                        bujus[index].state = CalcChildState::Finished;
                    }
                } else {
                    // 若没有等待局面，则稍作休眠，避免忙等
                    thread::sleep(Duration::from_millis(10));
                }
            }
        });
        handles.push(handle);
    }

    // 等待所有线程结束
    for handle in handles {
        handle.join().unwrap();
    }

    // 当解被找到时，通过 father 字段回溯整个解题过程
    let solution_index = {
        let sol = solution_found.lock().unwrap();
        sol.clone()
    };

    if let Some(sol_idx) = solution_index {
        // 回溯路径
        let bujus = BUJUS.lock().unwrap();
        let mut path = Vec::new();
        let mut cur_idx = sol_idx;
        loop {
            path.push(cur_idx);
            if let Some(father_idx) = bujus[cur_idx].father {
                cur_idx = father_idx;
            } else {
                break;
            }
        }
        path.reverse();
        println!("找到解法，共 {} 步", path.len());
        for idx in path {
            println!("局面 {}: {:?}", idx, bujus[idx]);
        }
    } else {
        println!("未找到解法");
    }
}
