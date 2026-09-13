# 所有性质搜索范围阈值
#相似
pp_diff_threshold = {"esol": 0.1, "caco2": 0.1,"ld50": 0.1, "lipo": 0.1, "freesolv": 1,
            "half_life": 10,"ppb": 20,"vdss": 20,"ch": 5, "cyp_2c9":1,"cyp_2d6":1,"cyp_3a4":1,
                "cl":0.1,"hum_mic_cl":0.1, "logd":0.1, "mou_mic_cl":0.1, "rat_mic_cl":0.1,"ppb":0.01,"water_sol": 0.1}
#相反
pp_diff = {"esol": 5, "caco2": 2,"ld50": 3, "lipo": 2, "freesolv": 5,
            "half_life": 10,"ppb": 20,"vdss": 20,"ch": 5, "cyp_2c9":50,"cyp_2d6":50,"cyp_3a4":50,
                "cl":5, "hum_mic_cl":5, "logd":5, "mou_mic_cl":5, "rat_mic_cl":5,"ppb":0.5,"water_sol": 5}

pp_des = {'HallKierAlpha':[-18, 8], 'MolWt':[16, 2600], 'NumAromaticRings':[0, 20], 'NumHAcceptors':[0, 50],
        'NumHDonors':[0, 50], 'NumRotatableBonds':[0, 150], 'TPSA':[0,966], 'MolLogP':[-20, 70], 'qed':[0, 1]}

pp_range = {"esol":[-13.1719,10], "caco2":[-7.76,-3.51],"ld50":[0.291,10.207], "lipo":[-1.5,4.5], "freesolv":[-25.47,3.43],
            "half_life":[0.065,69],"ppb":[10.09,99.95],"vdss":[0.01,60],"ch":[3,150],"cm":[3,150],
            "cyp_2c9":[0,100], "cyp_2d6":[0,100], "cyp_3a4":[0,100], "hum_mic_cl":[-3,7],
            "logd":[-5,5.2], "mou_mic_cl":[-3,7], "rat_mic_cl":[-3,7], "ppb":[0,1],"cl":[-3,7],
            "SA":[1,9],"water_sol": [-13.1719,10]}