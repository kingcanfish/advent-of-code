// 检查序列是否满足以下条件：
// 1. 单调递增或单调递减
// 2. 相邻元素差的绝对值在 1 到 3 之间
#[allow(unused)]
pub(crate) fn check(elements: &[i64]) -> bool {
    let diffs: Vec<i64> = elements
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();

    // 检查所有差值是否都在有效范围内且单调递增或递减
    diffs.iter().all(|&diff| diff.abs() >= 1 && diff.abs() <= 3)
        && (diffs.iter().all(|&diff| diff > 0) || diffs.iter().all(|&diff| diff < 0))
}

#[allow(unused)]
pub(crate) fn resolve(input: &[Vec<i64>]) -> i64 {
    input.iter().filter(|line| check(line)).count() as i64
}
