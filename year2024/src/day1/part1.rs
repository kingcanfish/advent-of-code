#[allow(unused)]
pub(crate) fn resolve(left: &[i64], right: &[i64]) -> i64 {
    let mut result = 0;

    // 创建可变副本并排序
    let mut sorted_left = left.to_owned();
    let mut sorted_right = right.to_owned();
    sorted_left.sort();
    sorted_right.sort();

    // 计算对应位置元素差的绝对值之和
    for i in 0..sorted_left.len() {
        result += (sorted_left[i] - sorted_right[i]).abs();
    }

    result
}
