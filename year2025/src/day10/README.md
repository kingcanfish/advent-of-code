# Day10: 灯泡机器 - 详细解题思路

## 目录
1. [问题理解](#问题理解)
2. [Part 1: GF(2) 线性方程组求解](#part-1-gf2-线性方程组求解)
3. [Part 2: 整数线性规划](#part-2-整数线性规划)
4. [核心算法详解](#核心算法详解)
5. [代码实现分析](#代码实现分析)
6. [复杂度分析](#复杂度分析)
7. [位运算优化详解](#位运算优化详解)
8. [Z3求解器详解](#z3求解器详解)

---

## 问题理解

### 题目背景
我们有一个控制机器，有：
- **N盏灯**：每盏灯可以是开(1)或关(0)的状态
- **M个按钮**：每个按钮控制一组灯，按下时会翻转这些灯的状态
- **目标状态**：希望最终达到的灯的开关模式

### 输入格式
```
[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
```

### 格式解析
```
[.##.]          - 目标模式：灯0关，灯1开，灯2开，灯3开
(3)             - 按钮0：控制灯3
(1,3)           - 按钮1：控制灯1和灯3  
(2)             - 按钮2：控制灯2
(2,3)           - 按钮3：控制灯2和灯3
(0,2)           - 按钮4：控制灯0和灯2
(0,1)           - 按钮5：控制灯0和灯1
{3,5,4,7}       - 目标值（Part2用）：灯0=3, 灯1=5, 灯2=4, 灯3=7
```

### 可视化示例
```
灯的状态表示：   . = 关(0), # = 开(1)
按钮控制关系：   按钮0 → 灯3
                按钮1 → 灯1, 灯3
                按钮2 → 灯2
                按钮3 → 灯2, 灯3
                按钮4 → 灯0, 灯2
                按钮5 → 灯0, 灯1

目标状态：       [.##.] = [0,1,1,1]
```

### 核心问题
**Part1**：每个按钮最多按一次（0或1次），找到最少的按钮组合使灯达到目标状态

**Part2**：每个按钮可以按任意次（0,1,2,3,...次），找到最少的总按压次数使灯达到目标值

---

## Part 1: GF(2) 线性方程组求解

### 核心概念

#### GF(2) - 二元域
GF(2) 是只包含 {0, 1} 两个元素的域，加法和乘法都在模2意义下进行：

```
加法（实际上是XOR操作）：
0 + 0 = 0
0 + 1 = 1  
1 + 0 = 1
1 + 1 = 0

乘法：
0 × 0 = 0
0 × 1 = 0
1 × 0 = 0
1 × 1 = 1
```

#### 为什么用GF(2)？
因为灯的开关状态是二进制的：
- 0 = 关，1 = 开
- 按钮按偶数次 = 没按
- 按钮按奇数次 = 按了一次

这正好符合GF(2)的运算规则！

### 数学建模

#### 变量定义
设 x₀, x₁, x₂, x₃, x₄, x₅ 表示每个按钮是否按下：
- xᵢ = 0：按钮i不按
- xᵢ = 1：按钮i按一次

#### 方程构建
对于每盏灯，写一个方程：所有控制它的按钮按压次数之和 ≡ 目标状态

以我们的例子：
```
灯0（目标=0）：按钮4和按钮5控制它
x₄ + x₅ ≡ 0 (mod 2)

灯1（目标=1）：按钮1和按钮5控制它  
x₁ + x₅ ≡ 1 (mod 2)

灯2（目标=1）：按钮2、按钮3、按钮4控制它
x₂ + x₃ + x₄ ≡ 1 (mod 2)

灯3（目标=1）：按钮0、按钮1、按钮3控制它
x₀ + x₁ + x₃ ≡ 1 (mod 2)
```

#### 增广矩阵表示
```
         x₀ x₁ x₂ x₃ x₄ x₅ | b
方程0:    0  0  0  0  1  1 | 0
方程1:    0  1  0  0  0  1 | 1  
方程2:    0  0  1  1  1  0 | 1
方程3:    1  1  0  1  0  0 | 1
```

### 解题步骤

#### 步骤1: 高斯消元（Forward Elimination）
目标是化为上三角矩阵：

```
消元前：
         x₀ x₁ x₂ x₃ x₄ x₅ | b
方程0:    0  0  0  0  1  1 | 0
方程1:    0  1  0  0  0  1 | 1  
方程2:    0  0  1  1  1  0 | 1
方程3:    1  1  0  1  0  0 | 1

找x₀的Pivot（方程3）：
交换方程0和方程3：

         x₀ x₁ x₂ x₃ x₄ x₅ | b
方程0:    1  1  0  1  0  0 | 1  ← Pivot
方程1:    0  1  0  0  0  1 | 1  
方程2:    0  0  1  1  1  0 | 1
方程3:    0  0  0  0  1  1 | 0

用方程0消去其他行的x₀：
方程1: (方程1不变，因为x₀系数已经是0)
方程2: (方程2不变，因为x₀系数已经是0)  
方程3: (方程3不变，因为x₀系数已经是0)

找x₁的Pivot（方程1）：
         x₀ x₁ x₂ x₃ x₄ x₅ | b
方程0:    1  1  0  1  0  0 | 1
方程1:    0  1  0  0  0  1 | 1  ← Pivot
方程2:    0  0  1  1  1  0 | 1
方程3:    0  0  0  0  1  1 | 0

用方程1消去方程0的x₁：
方程0 = 方程0 + 方程1：
         x₀ x₁ x₂ x₃ x₄ x₅ | b
方程0:    1  0  0  1  0  1 | 0  ← 更新后
方程1:    0  1  0  0  0  1 | 1
方程2:    0  0  1  1  1  0 | 1
方程3:    0  0  0  0  1  1 | 0

找x₂的Pivot（方程2）：
         x₀ x₁ x₂ x₃ x₄ x₅ | b
方程0:    1  0  0  1  0  1 | 0
方程1:    0  1  0  0  0  1 | 1
方程2:    0  0  1  1  1  0 | 1  ← Pivot
方程3:    0  0  0  0  1  1 | 0

找x₃的Pivot：没有（x₃列都是0，跳过）

找x₄的Pivot（方程3）：
         x₀ x₁ x₂ x₃ x₄ x₅ | b
方程0:    1  0  0  1  0  1 | 0
方程1:    0  1  0  0  0  1 | 1
方程2:    0  0  1  1  1  0 | 1
方程3:    0  0  0  0  1  1 | 0  ← Pivot
```

#### 步骤2: 检查一致性
检查是否有"0 = 1"的矛盾：
```
没有全0系数但右边为1的行 → 有解
```

#### 步骤3: 识别自由变量
x₃是自由变量（没有Pivot列）

#### 步骤4: 回代求解
先处理有Pivot的变量：

**x₃ = 0 时：**
```
x₃ = 0
从方程3: x₄ + x₅ = 0 → x₄ = x₅
从方程2: x₂ + x₃ + x₄ = 1 → x₂ + 0 + x₄ = 1 → x₂ = 1 + x₄
从方程1: x₁ + x₅ = 1
从方程0: x₀ + x₃ + x₅ = 0 → x₀ + 0 + x₅ = 0 → x₀ = x₅

设 x₅ = t (自由变量)
则: x₀ = t, x₁ = 1 + t, x₂ = 1 + t, x₃ = 0, x₄ = t
```

**x₃ = 1 时：**
```
x₃ = 1  
从方程3: x₄ + x₅ = 0 → x₄ = x₅
从方程2: x₂ + 1 + x₄ = 1 → x₂ + x₄ = 0 → x₂ = x₄
从方程1: x₁ + x₅ = 1
从方程0: x₀ + 1 + x₅ = 0 → x₀ + x₅ = 1 → x₀ = 1 + x₅

设 x₅ = t
则: x₀ = 1 + t, x₁ = 1 + t, x₂ = t, x₃ = 1, x₄ = t
```

#### 步骤5: 找到最优解
计算每种情况下的按钮按压次数：

**情况1 (x₃ = 0)：**
- t = 0: (0, 1, 1, 0, 0, 0) → 按3次
- t = 1: (1, 0, 0, 0, 1, 1) → 按3次

**情况2 (x₃ = 1)：**
- t = 0: (1, 1, 0, 1, 0, 0) → 按3次  
- t = 1: (0, 0, 1, 1, 1, 1) → 按4次

**最优解**：按3次（如 x₃=0, t=0: 按按钮1,2,5）

---

## Part 2: 整数线性规划

### 核心概念

#### 与Part1的区别
- **Part1**：每个按钮0或1次（二进制变量）
- **Part2**：每个按钮0,1,2,3,...次（非负整数变量）

#### 新的约束形式
不再只有开/关状态，而是有具体的数值目标：
```
按钮0按x₀次, 按钮1按x₁次, 按钮2按x₂次, ...
灯0的目标是3：x₄ + x₅ = 3
灯1的目标是5：x₁ + x₅ = 5  
灯2的目标是4：x₂ + x₃ + x₄ = 4
灯3的目标是7：x₀ + x₁ + x₃ = 7
```

### 数学建模

#### 决策变量
x₀, x₁, x₂, x₃, x₄, x₅ ≥ 0 (整数)

#### 约束条件
```
x₄ + x₅ = 3    (灯0目标)
x₁ + x₅ = 5    (灯1目标)
x₂ + x₃ + x₄ = 4  (灯2目标)
x₀ + x₁ + x₃ = 7  (灯3目标)
```

#### 目标函数
最小化总按压次数：minimize (x₀ + x₁ + x₂ + x₃ + x₄ + x₅)

### Z3求解器工作原理

#### 1. 约束编程范式
不是写算法，而是**声明式地描述问题**：

**传统算法思维**：
```
对于每个灯：
    尝试所有按钮组合
    检查是否满足目标
    找最优解
```

**约束编程思维**：
```
问题描述：
    - 变量x₀,x₁,... ≥ 0
    - 约束：x₄ + x₅ = 3, x₁ + x₅ = 5, ...
    - 目标：最小化 Σxᵢ
让求解器自动找到最优解
```

#### 2. Z3内部求解过程
Z3使用多种技术的组合：

**线性规划松弛**：
```
先忽略整数约束，求解线性规划
得到：x₀=2, x₁=3, x₂=1, x₃=4, x₄=1, x₅=2 (总计13次)
```

**割平面方法**：
```
添加切平面来逼近整数解
不断细化可行域
```

**分支定界**：
```
选择非整数变量x₀=2.5
分支1: x₀ ≤ 2
分支2: x₀ ≥ 3
分别求解两个子问题
```

**剪枝优化**：
```
如果当前最好解是13次
某分支的下界已经是14次
可以剪枝，停止搜索该分支
```

#### 3. 具体求解过程
```
初始问题：
minimize x₀+x₁+x₂+x₃+x₄+x₅
subject to:
    x₄ + x₅ = 3
    x₁ + x₅ = 5
    x₂ + x₃ + x₄ = 4
    x₀ + x₁ + x₃ = 7
    xᵢ ≥ 0, integer

Z3分析：
- 4个等式约束，6个变量 → 2个自由度
- 可以用2个变量表示其他4个变量

解的可能形式：
从约束可得：
x₅ = 3 - x₄
x₁ = 5 - x₅ = 2 + x₄  
x₃ = 7 - x₀ - x₁ = 5 - x₀ - x₄
x₂ = 4 - x₃ - x₄ = x₀ - 1

目标函数：
f = x₀ + x₁ + x₂ + x₃ + x₄ + x₅
  = x₀ + (2+x₄) + (x₀-1) + (5-x₀-x₄) + x₄ + (3-x₄)
  = x₀ + 9

要最小化f，需要最小化x₀，且满足：
x₀ ≥ 0, x₄ ≥ 0
x₁ = 2 + x₄ ≥ 0 (总是成立)
x₃ = 5 - x₀ - x₄ ≥ 0 → x₀ + x₄ ≤ 5
x₂ = x₀ - 1 ≥ 0 → x₀ ≥ 1

最优解：x₀ = 1, x₄ = 0
则：x₁ = 2, x₅ = 3, x₃ = 4, x₂ = 0
总按压次数 = 1+2+0+4+0+3 = 10
```

---

## 核心算法详解

### Part1: 位运算高斯消元

#### 为什么用位运算？

传统方法 vs 位运算方法：

**传统布尔矩阵方法**：
```rust
let mut matrix = vec![vec![false; n_buttons + 1]; n_lights];

// 设置系数
matrix[light_idx][button_idx] = true;

// XOR操作 - 需要循环
for c in 0..n_cols {
    matrix[row][c] ^= matrix[pivot_row][c];
}
```

**位运算方法**：
```rust
let mut rows = Vec::<u64>::new();

// 设置系数 - 一条语句
rows[light_idx] |= 1u64 << button_idx;

// XOR操作 - 一条语句
rows[row] ^= rows[pivot_row];
```

**性能提升**：
- 一个u64可以存储64个变量的系数
- 一次XOR操作同时处理64位
- 内存使用从O(n²)降到O(n)

#### 位运算详解

**1. 构造矩阵**
```rust
// 假设有6个按钮，4盏灯
let mut rows = vec![0u64; 4];  // rows[0], rows[1], rows[2], rows[3]

// 按钮0控制灯3: rows[3] |= 1 << 0
rows[3] = rows[3] | 0b000001  // 0b000001 = 1 << 0

// 按钮1控制灯1,3: rows[1] |= 1<<1, rows[3] |= 1<<1  
rows[1] = rows[1] | 0b000010  // 0b000010 = 1 << 1
rows[3] = rows[3] | 0b000001 | 0b000010 = 0b000011

// 最终rows[3] = 0b000011 表示按钮0和按钮1都控制灯3
```

**2. 消元过程**
```rust
// 假设当前行：rows[0] = 0b101011 (按钮0,1,3,5控制灯0)
// Pivot行：rows[1] = 0b001010 (按钮1,3控制灯1)

// 用XOR消元
rows[0] ^= rows[1];  
// 0b101011 XOR 0b001010 = 0b100001
// 消去了按钮1和按钮3的影响
```

**3. 目标位处理**
```rust
// 目标位在最高位 (bit n_buttons)
let target_bit = 1u64 << n_buttons;

// 检查目标：rows[i] & target_bit != 0 表示该灯目标为1
let target = (rows[i] & target_bit) != 0;

// 设置目标位：rows[i] |= target_bit
if target {
    rows[i] |= target_bit;
}
```

### Part2: Z3建模详解

#### 1. 变量创建
```rust
// 为每个按钮创建整数变量
let mut button_vars = Vec::new();
for i in 0..n_buttons {
    let var_name = format!("button_{}", i);
    let var = Int::new_const(&context, var_name.as_str());
    button_vars.push(var);
    
    // 添加非负约束
    optimizer.assert(&var.ge(&Int::from_i64(&context, 0)));
}
```

**变量命名**：
```
button_0, button_1, button_2, ...
```

#### 2. 约束构建
```rust
// 对于灯0，目标值=3，控制的按钮是[4,5]
// 需要约束：button_4 + button_5 = 3

let target_value = 3;
let button_indices = vec![4, 5];

let mut sum_ast = Int::from_i64(&context, 0);
for &btn_idx in &button_indices {
    sum_ast = sum_ast.add(&button_vars[btn_idx]);
}
optimizer.assert(&sum_ast._eq(&Int::from_i64(&context, target_value)));
```

#### 3. 目标函数
```rust
// 总按压次数 = button_0 + button_1 + ... + button_n
let mut total_presses = Int::from_i64(&context, 0);
for btn_var in &button_vars {
    total_presses = total_presses.add(btn_var);
}
optimizer.minimize(&total_presses);
```

#### 4. 求解过程
```rust
// 检查可满足性
match optimizer.check(&[]) {
    z3::SatResult::Sat => {
        // 获取模型
        let model = optimizer.get_model().unwrap();
        
        // 评估目标函数值
        let total_presses_value = model
            .eval(&total_presses, true)
            .unwrap()
            .as_i64()
            .unwrap() as usize;
            
        Some(total_presses_value)
    }
    _ => None,  // 不可满足
}
```

---

## 代码实现分析

### 完整代码结构

```rust
use crate::file::read_file;
use std::collections::HashMap;
use std::error::Error;
use std::ops::Add;
use z3::{Config, Context, Optimize};
use z3::ast::{Ast, Int};

// 1. 解析输入
fn parse_machine(line: &str) -> Machine {
    // 解析目标模式 [.##.]
    let pattern_start = line.find('[').unwrap() + 1;
    let pattern_end = line.find(']').unwrap();
    let target_pattern: Vec<bool> = line[pattern_start..pattern_end]
        .chars()
        .map(|c| c == '#')
        .collect();

    // 解析按钮 (3) (1,3) (2) ...
    let mut buttons = Vec::new();
    let mut pos = pattern_end;
    while let Some(start) = line[pos..].find('(') {
        let actual_start = pos + start + 1;
        let end = line[actual_start..].find(')').unwrap() + actual_start;
        let button_str = &line[actual_start..end];
        
        let button: Vec<usize> = button_str
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        buttons.push(button);
        pos = end + 1;
    }

    // 解析目标值 {3,5,4,7}
    let values_start = line.find('{').unwrap() + 1;
    let values_end = line.find('}').unwrap();
    let target_values: Vec<i64> = line[values_start..values_end]
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    Machine {
        target_pattern,
        buttons,
        target_values,
    }
}

// 2. Part1: 位运算求解
fn solve_part1_machine(machine: &Machine) -> Option<usize> {
    let n_buttons = machine.buttons.len();

    // 构造位矩阵
    let mut rows = Vec::<u64>::new();
    
    // 设置系数矩阵
    for button_idx in 0..n_buttons {
        for &light_idx in &machine.buttons[button_idx] {
            if rows.len() <= light_idx {
                rows.resize(light_idx + 1, 0);
            }
            rows[light_idx] |= 1u64 << button_idx;
        }
    }

    // 设置目标位
    for (light_idx, &target) in machine.target_pattern.iter().enumerate() {
        if rows.len() <= light_idx {
            rows.resize(light_idx + 1, 0);
        }
        if target {
            rows[light_idx] |= 1u64 << n_buttons;
        }
    }

    gaussian_elimination_bitwise(&mut rows, n_buttons)
}

// 3. Part1: 位运算高斯消元
fn gaussian_elimination_bitwise(rows: &mut [u64], n_vars: usize) -> Option<usize> {
    let n_rows = rows.len();
    let target_bit = 1u64 << n_vars;
    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();

    // 前向消元
    for col in 0..n_vars {
        let col_bit = 1u64 << col;
        
        // 找Pivot
        let mut pivot = None;
        for row in pivot_row..n_rows {
            if rows[row] & col_bit != 0 {
                pivot = Some(row);
                break;
            }
        }

        if let Some(pivot_idx) = pivot {
            // 交换行
            if pivot_idx != pivot_row {
                rows.swap(pivot_idx, pivot_row);
            }

            pivot_cols.push(col);

            // 消去其他行的该列
            for row in 0..n_rows {
                if row != pivot_row && (rows[row] & col_bit) != 0 {
                    rows[row] ^= rows[pivot_row];
                }
            }

            pivot_row += 1;
        }
    }

    // 检查一致性
    for &row in rows.iter() {
        let coeff_part = row & ((1u64 << n_vars) - 1);
        if coeff_part == 0 && (row & target_bit) != 0 {
            return None;
        }
    }

    // 找自由变量
    let mut pivot_vars = vec![false; n_vars];
    for &col in &pivot_cols {
        pivot_vars[col] = true;
    }
    let free_vars: Vec<usize> = (0..n_vars).filter(|&i| !pivot_vars[i]).collect();

    // 枚举自由变量找最优解
    let num_free = free_vars.len();
    let mut min_count = usize::MAX;

    for mask in 0..(1u64 << num_free) {
        let mut solution = vec![false; n_vars];

        // 设置自由变量
        for (i, &var_idx) in free_vars.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                solution[var_idx] = true;
            }
        }

        // 回代求解主变量
        for (row_idx, &col) in pivot_cols.iter().enumerate() {
            let row = rows[row_idx];
            let mut val = (row & target_bit) != 0;

            // 减去已知变量的贡献
            for (var_idx, &var_val) in solution.iter().enumerate() {
                if var_idx != col && var_val {
                    let var_bit = 1u64 << var_idx;
                    if (row & var_bit) != 0 {
                        val ^= true;
                    }
                }
            }
            solution[col] = val;
        }

        // 计算汉明重量
        let count = solution.iter().filter(|&&x| x).count();
        min_count = min_count.min(count);
    }

    Some(min_count)
}

// 4. Part2: Z3求解
fn solve_part2_machine(machine: &Machine) -> Option<usize> {
    let n_buttons = machine.buttons.len();

    // 创建Z3环境
    let config = Config::new();
    let context = Context::new(&config);
    let optimizer = Optimize::new(&context);

    // 创建按钮变量
    let mut button_vars = Vec::new();
    for i in 0..n_buttons {
        let var_name = format!("button_{}", i);
        button_vars.push(Int::new_const(&context, var_name.as_str()));
        optimizer.assert(&button_vars[i].ge(&Int::from_i64(&context, 0)));
    }

    // 构建按钮索引映射
    let mut button_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (btn_idx, button) in machine.buttons.iter().enumerate() {
        for &light_idx in button {
            button_indices
                .entry(light_idx)
                .or_insert_with(Vec::new)
                .push(btn_idx);
        }
    }

    // 添加约束
    for (&joltage_idx, indices) in &button_indices {
        if joltage_idx < machine.target_values.len() {
            let target_value = machine.target_values[joltage_idx];

            let mut sum_ast = Int::from_i64(&context, 0);
            for &btn_idx in indices {
                sum_ast = sum_ast.add(&button_vars[btn_idx]);
            }

            optimizer.assert(&sum_ast._eq(&Int::from_i64(&context, target_value)));
        }
    }

    // 最小化总按压次数
    let mut total_presses_ast = Int::from_i64(&context, 0);
    for btn_var in &button_vars {
        total_presses_ast = total_presses_ast.add(btn_var);
    }
    optimizer.minimize(&total_presses_ast);

    // 获取解
    match optimizer.check(&[]) {
        z3::SatResult::Sat => {
            let model = optimizer.get_model().unwrap();
            let total_presses = model
                .eval(&total_presses_ast, true)
                .unwrap()
                .as_i64()
                .unwrap() as usize;
            Some(total_presses)
        }
        _ => None,
    }
}

// 5. 驱动函数
fn resolve_part1(input: &str) -> Result<i64, Box<dyn Error>> {
    let machines = parse_input(input);
    let total: usize = machines.iter().filter_map(|m| solve_part1_machine(m)).sum();
    Ok(total as i64)
}

fn resolve_part2(input: &str) -> Result<i64, Box<dyn Error>> {
    let machines = parse_input(input);
    let total: usize = machines.iter().filter_map(|m| solve_part2_machine(m)).sum();
    Ok(total as i64)
}
```

### 关键函数解析

#### 1. `parse_machine`
**功能**：解析一行输入，提取机器配置

**解析步骤**：
```
输入: "[.##.] (3) (1,3) (2) {3,5,4,7}"

1. 目标模式: "[.##.]" → [false, true, true, true]
2. 按钮: "(3)" → [3]
          "(1,3)" → [1, 3]  
          "(2)" → [2]
3. 目标值: "{3,5,4,7}" → [3, 5, 4, 7]

输出: Machine {
    target_pattern: [false, true, true, true],
    buttons: [[3], [1, 3], [2]],
    target_values: [3, 5, 4, 7]
}
```

#### 2. `gaussian_elimination_bitwise`
**功能**：位运算实现GF(2)高斯消元

**核心技巧**：
- 用u64存储一行，bit i表示变量i的系数
- XOR操作同时处理整行
- 目标位在最高位

**示例追踪**：
```rust
// 初始矩阵（用二进制更直观）
// 变量: x0 x1 x2 x3 | 目标
// 行0:  0  0  0  1 | 1     (控制灯0的按钮: x3)
// 行1:  0  1  0  1 | 1     (控制灯1的按钮: x1, x3)  
// 行2:  0  0  1  1 | 1     (控制灯2的按钮: x2, x3)
// 行3:  1  1  1  0 | 0     (控制灯3的按钮: x0, x1, x2)

// 位表示 (4个变量，所以目标位在bit 4)
rows[0] = 0b10001  // x3 + 目标1
rows[1] = 0b01011  // x1 + x3 + 目标1  
rows[2] = 0b00111  // x2 + x3 + 目标1
rows[3] = 0b01110  // x0 + x1 + x2 + 目标0

// 找x0的pivot：rows[3]有x0
// 交换到第一行
// 消去其他行的x0：rows[3] = 0b01110
// rows[1] ^= rows[3] → 0b01011 XOR 0b01110 = 0b00101 (x1被消去)
```

#### 3. `solve_part2_machine`
**功能**：用Z3求解整数线性规划

**建模过程**：
```
问题: 3盏灯，4个按钮
按钮0→灯2, 按钮1→灯0,2, 按钮2→灯1, 按钮3→灯0,1,2
目标: 灯0=2, 灯1=1, 灯2=3

约束:
button_1 + button_3 = 2  (灯0)
button_2 + button_3 = 1  (灯1)  
button_0 + button_1 + button_3 = 3  (灯2)

minimize: button_0 + button_1 + button_2 + button_3

Z3求解: 可能得到 button_0=1, button_1=1, button_2=0, button_3=1
总按压次数 = 3
```

---

## 复杂度分析

### Part1: 位运算高斯消元

**时间复杂度**：
- 消元过程：O(n³/64)，其中n是变量个数
- 枚举自由变量：O(2^f × n²)，f是自由变量个数
- 实际表现接近O(n³/64)，比传统O(n³)快64倍

**空间复杂度**：
- 传统方法：O(n²) 个布尔值
- 位运算方法：O(n) 个u64
- 内存节省：64倍

**实际性能**：
```
n=64 (64个按钮): 
- 传统: ~262,000 次布尔操作
- 位运算: ~4,000 次64位操作
- 加速比: ~65倍

n=128 (128个按钮):
- 传统: ~2,097,000 次布尔操作  
- 位运算: ~16,000 次64位操作
- 加速比: ~130倍
```

### Part2: Z3整数线性规划

**时间复杂度**：
Z3的复杂度取决于具体问题：
- 线性情况：多项式时间
- 整数约束：NP难问题
- 实际表现：比指数级好很多

**空间复杂度**：
- 变量和约束的存储：O(n + m)
- Z3内部数据结构：O(n × m)

**实际性能**：
```
小规模 (变量<20): 毫秒级
中等规模 (变量<100): 秒级  
大规模 (变量>100): 可能需要分钟级
```

### 性能对比

| 方法 | 时间复杂度 | 空间复杂度 | 适用规模 |
|------|------------|------------|----------|
| Part1布尔矩阵 | O(n³) | O(n²) | n < 50 |
| Part1位运算 | O(n³/64) | O(n) | n < 1000 |
| Part2手工算法 | O(n³) | O(n²) | n < 20 |
| Part2 Z3 | 指数级(实际很好) | O(n+m) | n < 500 |

---

## 位运算优化详解

### 为什么位运算快？

#### 1. CPU级别的并行性
现代CPU一次可以处理64位数据：
```rust
// 传统方式：需要64次1位操作
for i in 0..64 {
    matrix[row][i] ^= matrix[pivot_row][i];
}

// 位运算：1次64位操作
rows[row] ^= rows[pivot_row];
```

#### 2. 内存访问优化
```rust
// 传统方式：64次内存读取
let mut row_data = vec![false; 65];
for i in 0..64 {
    row_data[i] = matrix[row][i];
}

// 位运算：1次内存读取
let row_data = rows[row];
```

#### 3. 缓存友好
64位整数更容易放入CPU缓存：
- L1缓存：32-64KB，可容纳512-1024个u64
- L2缓存：256KB-1MB，可容纳4096-16384个u64

### 位运算技巧详解

#### 1. 位掩码操作
```rust
// 检查第i位是否为1
if (value >> i) & 1 == 1 { ... }

// 设置第i位为1  
value |= 1 << i;

// 清除第i位
value &= !(1 << i);

// 翻转第i位
value ^= 1 << i;
```

#### 2. 批量操作
```rust
// 检查多个位
let has_any = value & mask != 0;     // 掩码中任意位为1
let has_all = (value & mask) == mask; // 掩码中所有位为1

// 清除多个位
value &= !mask;

// 提取连续位
let middle_bits = (value >> start) & ((1 << length) - 1);
```

#### 3. 高斯消元的位级实现
```rust
// 找pivot: 找某列中第一个1
let col_bit = 1u64 << col;
for row in pivot_row..n_rows {
    if rows[row] & col_bit != 0 {
        // 找到pivot
        break;
    }
}

// 消元: XOR整个行
for row in 0..n_rows {
    if row != pivot_row && (rows[row] & col_bit) != 0 {
        rows[row] ^= rows[pivot_row];  // 一次XOR完成整行消元
    }
}
```

### 性能测试示例

```rust
use std::time::Instant;

fn benchmark() {
    // 生成测试数据
    let n = 64;
    let mut rows = vec![0u64; n];
    
    // 填充随机数据
    for i in 0..n {
        rows[i] = rand::random::<u64>();
    }
    
    // 测试位运算版本
    let start = Instant::now();
    let result1 = gaussian_elimination_bitwise(&mut rows, n);
    let bitwise_time = start.elapsed();
    
    println!("位运算时间: {:?}", bitwise_time);
    println!("结果: {:?}", result1);
}
```

**预期结果**：
- n=64: < 1ms
- n=128: ~5ms  
- n=256: ~20ms
- n=512: ~100ms

---

## Z3求解器详解

### Z3架构概览

#### 1. 层次结构
```
┌─────────────────────────────────────┐
│         用户接口层                    │
│  (优化器、求解器、检查器)            │
├─────────────────────────────────────┤
│         核心求解引擎                  │
│  (SAT求解、LP求解、算法组合)         │
├─────────────────────────────────────┤
│         理论求解器                    │
│  (算术、数组、位向量、等等)          │
├─────────────────────────────────────┤
│         基础数据结构和算法            │
│  (图、哈希表、排序、搜索)            │
└─────────────────────────────────────┘
```

#### 2. 求解策略
Z3使用**portfolio方法**，同时运行多个求解策略：
```
策略1: 启发式搜索 (快速找到解)
策略2: 割平面法 (优化解的质量)  
策略3: 分支定界 (保证最优性)
策略4: 学习数据库 (避免重复搜索)
```

### 线性整数规划求解

#### 1. 预处理
```rust
// 问题：
minimize: x₀ + x₁ + x₂
subject to:
    2x₀ + x₁ = 5
    x₀ + 3x₂ = 7  
    xᵢ ≥ 0, integer

// Z3预处理：
// 1. 检测冗余约束
// 2. 变量替换：x₁ = 5 - 2x₀, x₂ = (7 - x₀)/3
// 3. 目标函数：x₀ + (5-2x₀) + (7-x₀)/3 = ...
```

#### 2. 松弛求解
```rust
// 忽略整数约束，求线性规划：
minimize: 8/3 - 4x₀/3  (x₀ ≥ 0)
subject to: x₀ ≤ 7 (从x₂ ≥ 0得到)

// 最优解：x₀ = 0, x₁ = 5, x₂ = 7/3 ≈ 2.33
// 目标值：0 + 5 + 2.33 = 7.33 (下界)
```

#### 3. 分支策略
```rust
// x₂ = 7/3 不是整数，选择x₂分支：

// 分支1: x₂ ≤ 2
//   重新求解：x₀ = 1, x₁ = 3, x₂ = 2
//   目标值：1 + 3 + 2 = 6

// 分支2: x₂ ≥ 3  
//   重新求解：不可行 (x₂ ≥ 3 → x₀ ≤ -2)
//   剪枝
```

#### 4. 最优解
```
最优解：x₀ = 1, x₁ = 3, x₂ = 2
目标值：6
```

### 高级功能

#### 1. 优化目标
```rust
// 多目标优化
optimizer.minimize(&cost);
optimizer.maximize(&profit);

// 权重优化  
let weighted_objective = cost.arithmetic_mul(profit_weight);
optimizer.minimize(&weighted_objective);
```

#### 2. 软约束
```rust
// 添加优先级的约束
let hard_constraint = x.gt(&Int::from_i64(&context, 0));
optimizer.assert(&hard_constraint);

let soft_constraint = y.lt(&Int::from_i64(&context, 10));
optimizer.assert_soft(&soft_constraint, 1); // 权重1
```

#### 3. 批量求解
```rust
let mut solver = Solver::new(&context);
for constraint in constraints {
    solver.assert(&constraint);
}

match solver.check() {
    SatResult::Sat => {
        let model = solver.get_model().unwrap();
        // 处理解
    }
    _ => println!("不可满足"),
}
```

### 性能调优

#### 1. 求解器选择
```rust
// 针对不同问题选择不同的求解器
let config = Config::new();
config.set_param("sat.random_seed", 42);      // SAT求解随机种子
config.set_param("timeout", 10000);          // 超时10秒
config.set_param("model.validate", true);    // 验证模型正确性

let context = Context::new(&config);
let optimizer = Optimize::new(&context);
```

#### 2. 建模技巧
```rust
// 好习惯：使用有意义的变量名
let x = Int::new_const(&context, "production_quantity");
let y = Int::new_const(&context, "storage_capacity");

// 避免冗余约束
// 坏：assert(x.gt(&y)); assert(x.gt(&y)); assert(x.gt(&y));
// 好：assert(x.gt(&y));

// 使用范围约束优化搜索
optimizer.assert(&x.ge(&Int::from_i64(&context, 0)));
optimizer.assert(&x.lt(&Int::from_i64(&context, 1000))); // 缩小搜索空间
```

#### 3. 监控求解过程
```rust
// 设置回调来监控进度
struct ProgressCallback {
    iterations: usize,
    best_bound: i64,
}

impl ast::Ast for ProgressCallback {
    fn get_id(&self) -> usize { self.iterations }
    // ... 其他方法
}

// 在求解过程中收集统计信息
let stats = optimizer.get_statistics();
println!("求解器统计: {:?}", stats);
```

---

## 总结与扩展

### Part1 关键要点
1. **GF(2)本质**：二进制域上的线性代数
2. **位运算优势**：64倍性能提升
3. **高斯消元**：通用的解线性方程组方法
4. **自由变量**：多个解时找最优解

### Part2 关键要点  
1. **整数规划**：比二进制问题更复杂
2. **Z3工具**：强大的约束求解器
3. **建模思维**：声明式vs命令式
4. **优化目标**：最小化/最大化函数

### 学习建议

#### 1. 数学基础
- **线性代数**：矩阵运算、向量空间
- **离散数学**：布尔代数、模运算
- **运筹学**：线性规划、整数规划

#### 2. 算法技能
- **高斯消元**：解线性方程组的标准方法
- **位运算**：性能优化的重要技巧
- **约束求解**：现代AI的重要工具

#### 3. 工具实践
- **Z3学习路径**：
  - 官方教程：https://z3prover.github.io/html/tutorial.html
  - 在线试用：https://rise4fun.com/z3
  - Python接口：更容易上手

- **位运算练习**：
  - 掌握基本操作：& | ^ ~ << >>
  - 算法题练习：LeetCode位运算专题
  - 性能分析：对比实现的时间复杂度

### 实际应用

#### 1. 电路设计
```rust
// 逻辑门优化：最小化门电路数量
// 约束：实现特定的布尔函数
// 目标：最小化面积或功耗
```

#### 2. 调度问题
```rust
// 任务分配：最小化总完成时间
// 约束：资源限制、依赖关系
// 目标：最小化Makespan
```

#### 3. 机器学习
```rust
// 特征选择：选择最有用的特征
// 约束：特征数量限制、准确率要求
// 目标：最大化模型性能
```

### 扩展思考

#### 1. 更复杂的约束
- 非线性约束
- 随机约束
- 动态约束

#### 2. 近似算法
- 当精确求解困难时的启发式方法
- 遗传算法、模拟退火
- 机器学习辅助的求解

#### 3. 并行化
- 多核CPU并行高斯消元
- GPU加速位运算
- 分布式约束求解

---

**希望这个详细文档能帮助你深入理解Day10的算法精髓！从基础的数学概念到高级的工具使用，逐步建立起完整的算法知识体系。**