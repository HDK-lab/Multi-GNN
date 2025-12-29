import csv

# 筛选活性值上升或不变的分子对数据
def filter_csv(input_file, output_file):
    try:
        with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 写入表头
            header = next(reader)
            writer.writerow(header)

            # 记录筛选出的行数
            filtered_rows = 0
            total_rows = 0

            # 逐行处理数据
            for row in reader:
                total_rows += 1
                try:
                    # 假设第三列和第四列是活性值
                    # col3 = float(row[1])
                    col4 = float(row[3])
                    if col4 > 0.6:
                        writer.writerow(row)
                        filtered_rows += 1
                except IndexError:
                    print(f"索引错误：行 {total_rows} 列数不足，内容为 {row}")
                    continue
                except ValueError:
                    print(f"值错误：行 {total_rows} 第三列或第四列无法转换为浮点数，内容为 {row}")
                    continue

            print(f"总共有 {total_rows} 行数据，筛选出 {filtered_rows} 行数据。")

    except FileNotFoundError:
        print(f"错误: 文件 {input_file} 未找到!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")

if __name__ == "__main__":
    input_file = 'D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\esol_bbbp_opt_data\ESOL_ALL_IS_sirt-FIN2_7_RES.csv'
    output_file = 'D:\PycharmProjects\FP_GNN_20240901\FP_GNN\\fpgnn\dataset\esol_bbbp_opt_data\\ESOL_ALL_IS_0.5_sirt-FIN2_7_RES.csv'
    filter_csv(input_file, output_file)