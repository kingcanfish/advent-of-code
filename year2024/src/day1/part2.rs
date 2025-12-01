#[allow(unused)]
pub(crate) fn resolve(left: &[i64], right: &[i64]) -> i64 {
    // 计算 左边的数字在右边出现了多少次 ；该次数与之相乘的和

    left.iter()
        .map(|&x| {
            let count = right.iter().filter(|&&y| x == y).count() as i64;
            count * x
        })
        .sum::<i64>()
}
