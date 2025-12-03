use std::error::Error;
use std::path::PathBuf;

pub(crate) fn read_input() -> Result<Vec<Vec<i64>>, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("inputs")
        .join("day3.txt");
    let content = std::fs::read_to_string(path)?;

    let result: Vec<Vec<i64>> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.chars()
                .filter(|c| c.is_ascii_digit())
                .map(|c| (c as u8 - b'0') as i64)
                .collect()
        })
        .collect();

    Ok(result)
}

pub(crate) fn resolve_part1(input: &[Vec<i64>]) -> i64 {
    input
        .iter()
        .map(|line| {
            let mut left = 0;
            let mut right = 0;

            for (i, &value) in line.iter().enumerate() {
                if i < line.len() - 1 && value > left {
                    left = value;
                    right = line[i + 1];
                    continue;
                }

                if value > right {
                    right = value
                }
            }
            left * 10 + right
        })
        .sum()
}

fn find_max_k_elements(k: usize, nums: &[i64]) -> i64 {
    let n = nums.len();

    // 使用二维数组，但只保留两行以节省空间
    // dp[0] 是前一行，dp[1] 是当前行
    let mut dp = vec![vec![0i64; k + 1]; 2];
    let mut prev_row = 0;
    let mut curr_row = 1;

    // 初始化：所有状态都为最小值，除了选0个
    dp[prev_row][0] = 0;
    dp[prev_row][1..=k].fill(i64::MIN);

    for (i, &num) in nums.iter().enumerate().take(n) {
        // 选0个始终是0
        dp[curr_row][0] = 0;

        // 从1到min(i+1, K)更新状态
        for j in 1..=std::cmp::min(i + 1, k) {
            // 不选当前元素
            dp[curr_row][j] = dp[prev_row][j];

            // 选当前元素（如果前一个状态有效）
            if dp[prev_row][j - 1] != i64::MIN {
                let new_value = dp[prev_row][j - 1].saturating_mul(10).saturating_add(num);
                dp[curr_row][j] = std::cmp::max(dp[curr_row][j], new_value);
            }
        }

        // 为下一次迭代交换行
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    // 返回最终结果
    dp[prev_row][k] // 注意：由于最后交换了，结果在prev_row中
}

pub fn resolve_part2(input: &[Vec<i64>]) -> i64 {
    input
        .iter()
        .map(|line| {
            if line.len() < 12 {
                return 0;
            }
            find_max_k_elements(12, line)
        })
        .sum()
}

pub(crate) fn run() {
    let input = read_input().unwrap();
    println!(
        "day 3 Part 1: {}, part2: {}",
        resolve_part1(&input),
        resolve_part2(&input)
    );
}
