import pandas as pd
import pandas_profiling

data = pd.read_csv(r'C:\Users\KDH\PycharmProjects\NLPStudy\resources\archive\spam.csv', encoding='latin1')

print(data[:5])

pr = data.profile_report()

pr.to_file('./resources/pr_report.html')
