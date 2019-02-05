from flocks_realtime import run_sim
#cnt = 1
#for diam in [50,100,250,500,100]:
#    for min_samples in [2, 5, 10, 20]:
#        print(f'### SIM #{cnt} OF 20 ###\n')
#        run_sim(diam = diam, min_samples = min_samples, min_card = 2)
#        cnt += 1

print('flocks')
run_sim(diam=500, min_samples=10, min_card=2, circular=True)
print('convoys')
run_sim(diam=100, min_samples=10, min_card=2, circular=False)
