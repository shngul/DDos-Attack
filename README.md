# DDos-Attack
Kali Linux sanal makinesi kullanarak DDoS saldırılarının simülasyonunu gerçekleştirip, oluşturulan veri seti üzerinde makine öğrenme algoritmaları ile saldırı tespiti ve normal trafikten ayırma.

Günümüzde bilgisayar ağlarında DDoS (Hizmet Reddi) saldırıları giderek artan bir tehdit haline gelmiştir. Bu saldırılar, bir saldırganın hedef ağı veya sunucuyu aşırı yüklemesiyle normal trafiğin engellenmesine yol açar, bu da hizmet kesintilerine ve ciddi maddi kayıplara neden olabilmektedir. Bu nedenle, DDoS saldırılarını tespit etmek ve bunlara yanıt vermek, ağ güvenliğinin önemli bir parçası haline gelmiştir.


Kali Linux sanal makinesi kullanılarak DDoS saldırılarının simülasyonu gerçekleştiridi ve ağ trafiği verisi toplandı. Öncelikle, Kali Linux sanal makinesi kuruldu ve ağ trafiğini kaydetmek için belirli komutlar çalıştırıldı. Bu işlemler sırasında, belirli bir hedef IP adresine sürekli SYN paketleri gönderilerek DDoS saldırısı başlatıldı ve kaydedilen ağ trafiği daha sonra analiz edilmek üzere saklandı. Python programlama dili kullanılarak, kaydedilen pcap dosyası okunup analiz edildi ve paketlerin çeşitli özellikleri çıkarılarak etiketlendi. Hazırlanan veri seti, DDoS saldırılarını tespit etmek ve normal trafikten ayırt etmek amacıyla farklı makine öğrenimi algoritmalarını eğitmek için kullanıldı. Veri setinin sütun başlıkları arasında IP kaynak ve hedef adresleri, TCP portları, IP protokolü, çerçeve uzunluğu, TCP bayrakları ve zaman damgası gibi bilgiler yer alırken, her paket saldırı veya normal trafik olarak etiketlendi. Bu kapsamlı ve etiketlenmiş veri seti, makine öğrenimi algoritmalarının etkinliğini değerlendirmek için kullanılarak, DDoS saldırılarını tespit etme ve normal trafikten ayırma konusunda başarılı sonuçlar elde edildi.


Veri setinde toplamda 30.000 veri örneği bulunmakta ve bu verilerden 10.000 tanesi "DDOS-ACK" etiketi ile işaretlenmişken, kalan 20.000 veri "Benign" etiketi ile sınıflandırıldı.


Çalışmamızda, Decision Tree (DT), Random Forest (RF), Support Vector Machine (SVM), Logistic Regression (LR), Gradient Boosting (GB), K-Nearest Neighbor (KNN) ve Yapay Sinir Ağları (YSA) gibi çeşitli sınıflandırma yöntemleri ve algoritmaları incelendi. Bu yöntemlerden hangisinin veri kümemize en uygun ve en etkili olduğunu belirlemek amacıyla kapsamlı bir analiz gerçekleştirildi. Bu analiz, projemizin amacına ulaşmak ve en doğru sonuçları elde etmek için kritik bir öneme sahiptir.


Sonuçta, DDoS saldırılarını tespit etme ve normal trafiğinden ayırma konusunda oldukça başarılı sonuçları Random Forest, Decision Tree, Gradient Boosting ve Logistic Regression algoritmaları elde etti.




ÖNEMLİ

Bu projede kullanılan veri seti, Kali Linux sanal makinesi üzerinde gerçekleştirilen DDoS saldırılarının simülasyonu ile elde edilmiştir. Ancak veri setinde bilgisayarın IP adresleri gibi hassas bilgiler bulunduğundan dolayı doğrudan paylaşılmamaktadır. Bunun yerine, veri setinin nasıl oluşturulacağını anlatan bir metin belgesi eklenmiştir.
