from register import main as register_main
from recognition import main as recognition_main

print("\n<FaceCard>\n")
print("무엇을 하시겠습니까?\n")
print("1. 등록")
print("2. 인식\n")
choice = input("번호선택: ").strip()

if choice == "1":
    register_main()
elif choice == "2":
    recognition_main()
else:
    print("잘못된 입력입니다.")