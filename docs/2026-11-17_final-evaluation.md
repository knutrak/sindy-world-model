# Final evaluation (pre-registered, run once)

Gates: `docs/2026-11-17_acceptance-gates.md` (FINAL). Script: `scripts/evaluate_final.py --final`.
Data: exp06 `final`, `final_cf_*`; exp07 `final`. Every number: `data/processed/exp06_intents/final_evaluation.json`.

## Gates and verdicts

```
                             A          W                          B                           S
G1                        True       True                       True                        True
G2                        True       True                       True                       False
G3                       False      False                       True                        True
G4                        True       True                       True                        True
G5                        True       True                       True                        True
G7                        True       True                       True                        True
G8                        True       True                       True                        True
G6 horizon final             2          2                          3                           4
G6 horizon final_cf          2          2                          4                           4
verdict              not valid  not valid  valid interpretable model  valid predictive extension
transfers                False      False                       True                        True
```

## A

- G1 term counts: {'health_blue': 1, 'health_red': 2, 'distance': 5, 'alive_blue': 1, 'alive_red': 2}
- G2 sign failures: none
- G3a terms failing the bootstrap rule: none
- G3b: {'same term set': False, 'term sets that differ': {'SG 17': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1'], 'alive_red': ['1', 'focus', 'wd']}, 'weak 1 s': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1', 'focus'], 'alive_red': ['1', 'focus']}, 'weak 2 s': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1', 'focus'], 'alive_red': ['1', 'focus']}}, 'health coefficients': {'health_blue: Rr': {'values': [-0.1863, -0.177, -0.1742, -0.1824, -0.184], 'max deviation from median': 0.04510484180919867}, 'health_red: Rb': {'values': [-0.1934, -0.1861, -0.1832, -0.1923, -0.1944], 'max deviation from median': 0.04740564827811566}, 'health_red: Rb*wd': {'values': [0.1783, 0.1633, 0.1541, 0.1733, 0.1698], 'max deviation from median': 0.09281789574895509}}}
- G4: {'final': {'finite': True, 'non-negative': True, 'alive never above start': True}, 'final_cf': {'finite': True, 'non-negative': True, 'alive never above start': True}}
- G5 (paired 95% intervals): {'model − baseline, health 4 s': {'persistence': [-0.1730220724317571, -0.15461218099420349], 'Lanchester': [-0.051537642565500766, -0.044104014189146605], 'no control': [-0.029544513769042064, -0.023656401383201582]}}
- G6: {'final': {'1 s': {'margin R2': 0.44697075804446806, 'interval': [0.4205781874976925, 0.4722108864051795], 'meets rule': True}, '2 s': {'margin R2': 0.3845512794878301, 'interval': [0.31960473083984137, 0.4402707107604199], 'meets rule': True}, '3 s': {'margin R2': 0.21230084380646475, 'interval': [0.10020494386824756, 0.3043103755544117], 'meets rule': False}, '4 s': {'margin R2': 0.023047093710799493, 'interval': [-0.14746627834615988, 0.15800922672916434], 'meets rule': False}, '6 s': {'margin R2': -0.3012324821050416, 'interval': [-0.5178548718440612, -0.1180961362570023], 'meets rule': False}, '8 s': {'margin R2': -0.6413392286642858, 'interval': [-0.9632734349058749, -0.38308846354889287], 'meets rule': False}}, 'final_cf': {'1 s': {'margin R2': 0.48206082187629484, 'interval': [0.457543079818711, 0.505629768486601], 'meets rule': True}, '2 s': {'margin R2': 0.4376095665523041, 'interval': [0.3923012477648121, 0.48181072063347025], 'meets rule': True}, '3 s': {'margin R2': 0.29440592646957575, 'interval': [0.22343640320353086, 0.3613378914591705], 'meets rule': False}, '4 s': {'margin R2': 0.16629083758037266, 'interval': [0.07881267518428824, 0.24534329113049472], 'meets rule': False}, '6 s': {'margin R2': -0.17971605643799626, 'interval': [-0.31655797286204734, -0.057595729509531984], 'meets rule': False}, '8 s': {'margin R2': -0.7041441199630947, 'interval': [-0.9420565208465485, -0.4925566991784117], 'meets rule': False}}}
- G7: {'health 4 s held-out': np.float64(0.05356042616236768), 'health 4 s val': np.float64(0.05510750738744101), 'ratio': np.float64(0.9719261258870528)}
- G8: {'model − baseline, stalkers health 4 s': {'persistence': [-0.04484168946454848, -0.03758696559895553], 'no control': [-0.019859822651133167, -0.016002969096987268]}}

## W

- G1 term counts: {'health_blue': 1, 'health_red': 2, 'distance': 5, 'alive_blue': 2, 'alive_red': 2}
- G2 sign failures: none
- G3a terms failing the bootstrap rule: none
- G3b: {'same term set': False, 'term sets that differ': {'SG 17': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1'], 'alive_red': ['1', 'focus', 'wd']}, 'weak 1 s': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1', 'focus'], 'alive_red': ['1', 'focus']}, 'weak 2 s': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1', 'focus'], 'alive_red': ['1', 'focus']}}, 'health coefficients': {'health_blue: Rr': {'values': [-0.1863, -0.177, -0.1742, -0.1824, -0.184], 'max deviation from median': 0.04510484180919867}, 'health_red: Rb': {'values': [-0.1934, -0.1861, -0.1832, -0.1923, -0.1944], 'max deviation from median': 0.04740564827811566}, 'health_red: Rb*wd': {'values': [0.1783, 0.1633, 0.1541, 0.1733, 0.1698], 'max deviation from median': 0.09281789574895509}}}
- G4: {'final': {'finite': True, 'non-negative': True, 'alive never above start': True}, 'final_cf': {'finite': True, 'non-negative': True, 'alive never above start': True}}
- G5 (paired 95% intervals): {'model − baseline, health 4 s': {'persistence': [-0.17360336913262217, -0.1553493563045841], 'Lanchester': [-0.05231208576934585, -0.04455708110028519], 'no control': [-0.03029140302840107, -0.02407543818326319]}}
- G6: {'final': {'1 s': {'margin R2': 0.42309557228737305, 'interval': [0.3956764271354292, 0.4495168480415064], 'meets rule': True}, '2 s': {'margin R2': 0.34620768747393793, 'interval': [0.2768311799063445, 0.4041038188173256], 'meets rule': True}, '3 s': {'margin R2': 0.1657248190784889, 'interval': [0.047545722808809705, 0.261880853838751], 'meets rule': False}, '4 s': {'margin R2': -0.02756689681128166, 'interval': [-0.1901505506971866, 0.10566274847408573], 'meets rule': False}, '6 s': {'margin R2': -0.37998662558634333, 'interval': [-0.6095341767563819, -0.19147786051273297], 'meets rule': False}, '8 s': {'margin R2': -0.7616993899364894, 'interval': [-1.1003555624577581, -0.48903309446099674], 'meets rule': False}}, 'final_cf': {'1 s': {'margin R2': 0.46099230347497433, 'interval': [0.43563145424373656, 0.4858251757437691], 'meets rule': True}, '2 s': {'margin R2': 0.40572159136802366, 'interval': [0.3585418340727092, 0.45161065679179163], 'meets rule': True}, '3 s': {'margin R2': 0.2606625222784633, 'interval': [0.1890233812880539, 0.3287418194372704], 'meets rule': False}, '4 s': {'margin R2': 0.1269659302090066, 'interval': [0.03818072424534056, 0.20990358347858418], 'meets rule': False}, '6 s': {'margin R2': -0.24748059459100968, 'interval': [-0.3998601070638347, -0.11500744451659774], 'meets rule': False}, '8 s': {'margin R2': -0.8230000368229602, 'interval': [-1.085879694953421, -0.5924318461830316], 'meets rule': False}}}
- G7: {'health 4 s held-out': np.float64(0.05294142740168425), 'health 4 s val': np.float64(0.05422135696223781), 'ratio': np.float64(0.9763943650203192)}
- G8: {'model − baseline, stalkers health 4 s': {'persistence': [-0.04463428567633613, -0.03751693181002996], 'no control': [-0.019829986914898463, -0.01586404106973077]}}
- margin R² at 4 s, model − A (paired 95%): [-0.06306617963090257, -0.037037227918069425]

## B

- G1 term counts: {'health_blue': 1, 'health_red': 2, 'distance': 5, 'alive_blue': 1, 'alive_red': 2}
- G2 sign failures: none
- G3a terms failing the bootstrap rule: none
- G4: {'final': {'finite': True, 'non-negative': True, 'alive never above start': True}, 'final_cf': {'finite': True, 'non-negative': True, 'alive never above start': True}}
- G5 (paired 95% intervals): {'model − baseline, health 4 s': {'persistence': [-0.17630574919121866, -0.15865713839158122], 'Lanchester': [-0.055365551849408674, -0.04754947665156538], 'no control': [-0.03321862380866772, -0.026967486568659733]}}
- G6: {'final': {'1 s': {'margin R2': 0.48249574420166796, 'interval': [0.46103628443810424, 0.5036324221064471], 'meets rule': True}, '2 s': {'margin R2': 0.49846144777676804, 'interval': [0.47013534913400673, 0.5257908961671182], 'meets rule': True}, '3 s': {'margin R2': 0.40461855797347823, 'interval': [0.3637536494543371, 0.44511889691755124], 'meets rule': True}, '4 s': {'margin R2': 0.27510426494132456, 'interval': [0.21647064092459434, 0.3297727466259759], 'meets rule': False}, '6 s': {'margin R2': -0.05039994699689343, 'interval': [-0.1670473392548689, 0.04698975345981706], 'meets rule': False}, '8 s': {'margin R2': -0.40020740603074967, 'interval': [-0.6046895298090567, -0.24673019617407], 'meets rule': False}}, 'final_cf': {'1 s': {'margin R2': 0.5127421292529483, 'interval': [0.49012598585931605, 0.533076483107629], 'meets rule': True}, '2 s': {'margin R2': 0.5411327845037321, 'interval': [0.5125297851422835, 0.5676557886087348], 'meets rule': True}, '3 s': {'margin R2': 0.46468521077028824, 'interval': [0.4268266511670193, 0.5000789703226527], 'meets rule': True}, '4 s': {'margin R2': 0.3580225462751154, 'interval': [0.30666615403681935, 0.40471317881689567], 'meets rule': True}, '6 s': {'margin R2': 0.035465615456657495, 'interval': [-0.06122531052489591, 0.12667198487491754], 'meets rule': False}, '8 s': {'margin R2': -0.4477335662501998, 'interval': [-0.6322084822707629, -0.28077609726735264], 'meets rule': False}}}
- G7: {'health 4 s held-out': np.float64(0.04990988012481397), 'health 4 s val': np.float64(0.05182837764704716), 'ratio': np.float64(0.9629836470032264)}
- G8: {'model − baseline, stalkers health 4 s': {'persistence': [-0.053477858418737666, -0.04600488590841894], 'no control': [-0.02850969936284851, -0.024501456334635897]}}
- margin R² at 4 s, model − A (paired 95%): [0.14363844109242643, 0.4119888732270315]

## S

- G1 term counts: {'health_blue': 1, 'health_red': 2, 'distance': 5, 'alive_blue': 1, 'alive_red': 2, 'share_blue': 5, 'share_red': 6}
- G2 sign failures: ['share_red: sf_adv = -0.184 (wrong sign)', 'share_red: sf_hold = -0.159 (wrong sign)']
- G3a terms failing the bootstrap rule: none
- G4: {'final': {'finite': True, 'non-negative': True, 'alive never above start': True}, 'final_cf': {'finite': True, 'non-negative': True, 'alive never above start': True}}
- G5 (paired 95% intervals): {'model − baseline, health 4 s': {'persistence': [-0.17773904162287024, -0.16009119797691537], 'Lanchester': [-0.05666572826069419, -0.04889191251625553], 'no control': [-0.034603257939134324, -0.028452262369676656]}}
- G6: {'final': {'1 s': {'margin R2': 0.5028046430820152, 'interval': [0.4820022993589161, 0.5221249534643376], 'meets rule': True}, '2 s': {'margin R2': 0.5528578804978795, 'interval': [0.5285359712141124, 0.5742257246722607], 'meets rule': True}, '3 s': {'margin R2': 0.4905743118616065, 'interval': [0.45807579045089736, 0.5197474647464394], 'meets rule': True}, '4 s': {'margin R2': 0.3993075890755653, 'interval': [0.3553232936876762, 0.4386614792431525], 'meets rule': True}, '6 s': {'margin R2': 0.1790899819729035, 'interval': [0.09632187467003268, 0.24939657161573428], 'meets rule': False}, '8 s': {'margin R2': -0.04439394108274741, 'interval': [-0.1833263692998305, 0.06224632808729183], 'meets rule': False}}, 'final_cf': {'1 s': {'margin R2': 0.5333590947421132, 'interval': [0.5122929420507625, 0.5525459374875233], 'meets rule': True}, '2 s': {'margin R2': 0.5924171910193805, 'interval': [0.5673622242047685, 0.6146524553405642], 'meets rule': True}, '3 s': {'margin R2': 0.546358821490457, 'interval': [0.5165733740855984, 0.5755624175291837], 'meets rule': True}, '4 s': {'margin R2': 0.47497281260319, 'interval': [0.4361723510891911, 0.5099441235903582], 'meets rule': True}, '6 s': {'margin R2': 0.26286958565785057, 'interval': [0.19913363435213374, 0.32435449120436466], 'meets rule': False}, '8 s': {'margin R2': -0.057041985253223126, 'interval': [-0.18551478267590485, 0.05604307997018129], 'meets rule': False}}}
- G7: {'health 4 s held-out': np.float64(0.04844826237524482), 'health 4 s val': np.float64(0.05021162656482111), 'ratio': np.float64(0.9648813569642111)}
- G8: {'model − baseline, stalkers health 4 s': {'persistence': [-0.05705240411417803, -0.04983253390521139], 'no control': [-0.032374048464501565, -0.02789220723273538]}}
- margin R² at 4 s, model − A (paired 95%): [0.2578206017638767, 0.5321869017758694]
