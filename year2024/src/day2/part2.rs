// Part 2 的检查函数：
// 允许移除一个元素，如果移除后序列满足 part1 的检查条件则返回 true
#[allow(unused)]
pub(crate) fn check(elements: &[i64]) -> bool {
    // 如果原本就满足条件，直接返回 true
    if crate::day2::part1::check(elements) {
        return true;
    }
    
    // 尝试移除每一个可能的元素
    (0..elements.len()).any(|i| {
        let new_elements = [&elements[..i], &elements[i+1..]].concat();
        crate::day2::part1::check(&new_elements)
    })
}

// 解析函数 - 统计满足条件的序列数量
#[allow(unused)]
pub(crate) fn resolve(input: &[Vec<i64>]) -> i64 {
    input.iter().filter(|line| check(line)).count() as i64
}
