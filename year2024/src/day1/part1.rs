use std::fs::File;
use std::io::{BufRead, BufReader, Result};

#[allow(unused)]
pub(crate) fn read_input(file_name: &str) -> Result<(Vec<i64>, Vec<i64>)> {
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);

    reader
        .lines()
        .try_fold((Vec::new(), Vec::new()), |(mut left, mut right), line| {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() >= 2 {
                let left_num: i64 = parts[0]
                    .parse()
                    .map_err(|_| invalid_data("Invalid number in left column"))?;
                let right_num: i64 = parts[1]
                    .parse()
                    .map_err(|_| invalid_data("Invalid number in right column"))?;

                left.push(left_num);
                right.push(right_num);
            }

            Ok((left, right))
        })
}

#[allow(unused)]
fn invalid_data(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}
#[allow(unused)]
pub(crate) fn resolve(left: Vec<i64>, right: Vec<i64>) -> i64 {
    let mut result = 0;

    // 创建可变副本并排序
    let mut sorted_left = left.clone();
    let mut sorted_right = right.clone();
    sorted_left.sort();
    sorted_right.sort();

    // 计算对应位置元素差的绝对值之和
    for i in 0..sorted_left.len() {
        result += (sorted_left[i] - sorted_right[i]).abs();
    }

    result
}
