import cv2
from pypylon import pylon
import time
import os

def basler_uzun_mp4_kaydet(kayit_suresi_sn=90):
    hedef_klasor = r"C:\BaslerKayitlar"
    if not os.path.exists(hedef_klasor):
        os.makedirs(hedef_klasor)
    
    dosya_yolu = os.path.join(hedef_klasor, "basler_90sn_kayit.mp4")

    # connect to cameraa
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    
    if not devices:
        print("Basler kamera bulunamadı! USB bağlantısını kontrol edin.")
        return

    camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
    camera.Open()
    print(f"Bağlanılan Kamera: {camera.GetDeviceInfo().GetFriendlyName()}")

    # start grabbing imagesd
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    # automatic resolution 
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if not grab_result.GrabSucceeded():
        print("Kameradan görüntü alınamadı.")
        return
    
    genislik = grab_result.Width
    yukseklik = grab_result.Height
    grab_result.Release()

    # 3. MP4 Video converter
    fps = 30.0  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_yazici = cv2.VideoWriter(dosya_yolu, fourcc, fps, (genislik, yukseklik))

    # Basler format to OpenCV format
    donusturucu = pylon.ImageFormatConverter()
    donusturucu.OutputPixelFormat = pylon.PixelType_BGR8packed

    print(f"\n[BAŞLADI] {kayit_suresi_sn} saniyelik (1 dk 30 sn) MP4 video kaydediliyor...")
    print(f"Kayıt yeri: {dosya_yolu}")
    
    baslangic_zamani = time.time()
    toplam_kare = 0
    son_yazdirilan_sn = -1

    try:
        
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grab_result.GrabSucceeded():
                donusmus_goruntu = donusturucu.Convert(grab_result)
                img = donusmus_goruntu.GetArray()

                video_yazici.write(img)
                toplam_kare += 1

                # time check
                gecen_sure = time.time() - baslangic_zamani
                kalan_sure = max(0.0, kayit_suresi_sn - gecen_sure)

                # time remaining 
                if int(gecen_sure) != son_yazdirilan_sn:
                    print(f"Kayıt ediliyor... Kalan Süre: {int(kalan_sure)} saniye", end='\r')
                    son_yazdirilan_sn = int(gecen_sure)

                # time remaining output
                sayaç_metni = f"Kalan: {int(kalan_sure)}sn / Toplam: {kayit_suresi_sn}sn"
                cv2.putText(img, sayaç_metni, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # live overview tab
                cv2.imshow('Basler 1 Dakika 30 Saniye Kayit', img)
                
                # exit with q 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nKullanıcı kaydı manuel olarak durdurdu (q tuşu).")
                    break
                
                # after 90 seconds = quit
                if gecen_sure >= kayit_suresi_sn:
                    print(f"\nSüre doldu ({kayit_suresi_sn} sn). Kayıt başarıyla tamamlandı.")
                    break
            
            grab_result.Release()
            
    except Exception as e:
        print(f"\nKayıt esnasında bir hata oluştu: {e}")
        
    finally:
        # close and save
        camera.StopGrabbing()
        camera.Close()
        video_yazici.release() 
        cv2.destroyAllWindows()
        print(f"\n[BİTTİ] Toplam {toplam_kare} kare tek bir MP4 dosyasında birleştirildi.")
        print(f"Videonuz burada hazır: {dosya_yolu}")

if __name__ == "__main__":
    
    basler_uzun_mp4_kaydet(kayit_suresi_sn=90)