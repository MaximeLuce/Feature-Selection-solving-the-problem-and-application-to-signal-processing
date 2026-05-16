from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151) 
  
# data (as pandas dataframes) 
X = connectionist_bench_sonar_mines_vs_rocks.data.features 
y = connectionist_bench_sonar_mines_vs_rocks.data.targets 
  
# metadata 
print(connectionist_bench_sonar_mines_vs_rocks.metadata) 
  
# variable information 
print(connectionist_bench_sonar_mines_vs_rocks.variables) 
