from matplotlib import pyplot as plt


def plot_sota_acc_trend(df, out: str = None):
    # 计算每一类的平均精度
    df['A_avg'] = df[['1', '2', '3']].mean(axis=1)
    df['B_avg'] = df[['4', '5', '6']].mean(axis=1)
    df['C_avg'] = df[['7', '8', '9']].mean(axis=1)

    # 绘制每一类的平均精度图
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['A_avg'], marker='o', label='FedAvg')
    plt.plot(df.index, df['B_avg'], marker='x', label='Scaffold')
    plt.plot(df.index, df['C_avg'], marker='^', label='Moon')
    plt.title('Average Accuracy of Each Class Over Time')
    plt.xlabel('Time/Index')
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    if out:
        plt.savefig(out)
    plt.show()
