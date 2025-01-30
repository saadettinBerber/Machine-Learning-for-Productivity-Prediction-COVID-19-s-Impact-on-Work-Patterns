# Machine-Learning-for-Productivity-Prediction-COVID-19-s-Impact-on-Work-Patterns

Hayatımıza Covid-19 hastalığı girdiğinde dünya düzeninde çok büyük değişimler oldu. En büyük değişiminden nasibii alan alanlardan biri tabi iş hayatı oldu. Birçok insan evden çalışmak zorunda kaldı ya da bazı meslekler haftanın bazı günleri işe gidebildi. Bizim veri setimiz ise değişen iş koşullarına karşılık insanların hayatına etkisi nasıl olduğuna dair bilgiler içermektedir. İnsan hayatına etkisini 15 tane özellik(sütun) ile ifade etmişler ve veri seti setinde 10.000 tane örnek kişi bulunmaktadır. Özellikleri incelediğimiz zaman:

- Increased_Work_Hours: Pandemi ile birlikte mesai saati artmışsa 1 ile ifade edilmiş ya da değişmemişse ve azalmışsa 0 ile ifade edilmiş.
-  Work_From_Home: Pandemi ile birlikte evden çalışma düzenine geçilmişse 1 ile ifade edilir geçilmemişse 0 ile ifade edilir.
-  Hours_Worked_Per_Day: İnsanın günde ne kadar çalıştığını ifade eden sayısal bir değer.
-  Meetings_Per_Day: İnsanların günde kaç saat online toplantılara girdiğini ifade eden sayısal değer
-  Productivity_Change: Pandemi nedeni ile insanların üretkenliği değişmişse 1 ile ifade edilir değişmemişse 0 ile ifade edilir.
-  Stress_Level: Pandemi ile birlikte insanların stres seviyeleri ifade edilmiştir(Low, Medium, High)
-  Health_Issue: Pandemi ile birlikte insanların yeni sağlık sorunları ortaya çıkmışsa 1 ile, değişmemişse 0 ile edilmiştir.
-  Job_Security: Pandemiden sonra bireylerin çalışırken kendini daha az güvende hissediyorsa 1 ile etiketlenmiştir.
-  Childcare_Responsibilities: Pandemiden sonra çocuk bakımı artışı 1 ile, değişim yok veya azalmışa 0 ile ifade edilmiştir.
-  Commuting_Changes: İnsanların işe gidip gelme alışkanlıkları değişmişse 1 ile değişmemişse 0 ile ifade edilir.
-  Technology_Adaptation: İnsanların uzaktan çalışabilmesi için teknolojilere uyum sağlaması gerektiği 1 ile ifade edilmiştir.
-  Salary_Changes: Pandemide maaş değişiklikleri olmuşsa 1 ile değişmemişse 0 ile ifade edilir.
-  Team_Collaboration_Challenges: Pandemi sırasında takım çalışmalarında insanlar zorlanması 1 ile ifade edilmesidir.
-  Sector: Bireyin hangi sektörde çalıştığını ifade eder.
  -Healthcare
  -IT
  -Education
  -Retail
-Affected_by_Covid: İnsan Covid-1'den kaynaklı olarak iş hayatı etkilenmişse 1 olarak ifade edilmiş.
