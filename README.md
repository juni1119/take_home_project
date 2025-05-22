# take_home_project

논문을 다 이해한 이후, 코드를 한 줄씩 읽어보았습니다. 

주어진 issue가 pretrained weight를 활용한 inference 과정에 문제가 발생한 것이기 때문에 infer.py 코드를 중심으로 읽어보았습니다.

초반 세팅 부분 이후, model 코드부터 복잡함을 느꼈습니다. 

DDPM object를 생성하면, 인스턴스 변수 netG를 정의하는 과정에서 unet model이 정의되고, unet model과 gaussian diffusion model이 만들어집니다.
그리고 test 단계에서는 stage 1에 해당하는 netG network와 stage 2에 해당하는 netH network가 사전 학습된 weight를 활용해 condition(J, A, trmap)과 최종 output(SR)을 만들어냅니다.

이후에 diffusion.py와 unet.py를 깊이 공부하는 과정에서 attention과 positional encoding 등에서 이론적으로 공부가 덜 된 상태임을 인지하였습니다. 그래서 gpt가 만들어준 기본적인 unet의 코드를 읽으면서 공부했고, DehazeDDPM에서 unet이 어떻게 역할을 수행하는지 구체적으로 감을 잡을 수 있었습니다.


그 이후에 해당 코드를 실제로 실행시켜 저에게도 pretrained weight를 활용한 inference 과정에 문제가 생김을 확인했습니다. 몇 가지 시도를 해보았지만 여전히 noise가 제거되지 않았고, 반복되는 시도 끝에 면접 준비를 마무리하게 되었습니다.

한 이미지 샘플을 만들어내는데 시간이 많이 들어 테스트 해보는 것이 다소 힘들었습니다. 그리고 그 과정에서 연산량이 많다보니 저에게 주어진 colab 사용량이 초과되어 코드를 실행시키는데 어려움도 있었습니다.

만약 저에게 시간이 조금 더 주어진다면, 이론적인 부분을 더 공부해서 코드를 이해하는데 집중할 것 같습니다. model이 어떻게 작동하는지 정확하게 파악이 된다면 디버깅을 해낼 수 있을거란 생각이 들어 아쉬웠습니다.




첫 번째 시도.

dataset을 다운받아 곧바로 infer.py를 실행시켜 보았습니다.
깃허브 이슈에 올라온 문제와 동일한 증상이 저에게도 발견되었습니다.

![try_1](https://github.com/juni1119/take_home_project/raw/main/try_1.png)

두 번째 시도.

코드를 읽다보니 config에서 stage 2의 pretrained_weight를 잘못된 경로로 사용하고 있다는 것을 알게 되었습니다. github에서 해당 weight 파일을 다운받아 stage 1과 stage 2에 정확한 경로를 지정해주었습니다. 하지만 여전히 noise 문제는 해결되지 않았습니다.

![try_1](https://github.com/juni1119/take_home_project/raw/main/try_2.png)

세 번째 시도.

이 과정에서 최종 이미지 하나를 출력하는데 1시간 30분이 걸렸고, 그 과정을 줄이고자 했습니다. 그래서 샘플링 수와 배치 사이즈를 대폭 줄였습니다. 이후 10분 가량으로 출력 시간이 단축되었지만, 여전히 노이즈가 출력되는 문제가 해결되지 않았습니다.

![try_1](https://github.com/juni1119/take_home_project/raw/main/try_3.png)

네 번째 시도.

그 이후에 stage 1이 만들어내는 condition이 어떻게 출력되는지 확인한다면, stage 1과 stage 2 중 어디에서 문제가 발생했는지 알 수 있을거라 생각했습니다. 그래서 J와 trmap 이미지를 저장할 수 있도록 self.output과 self.out_T를 save_image 해보았습니다. 그 때 아래와 같이 비교적 정상적인 이미지가 만들어짐을 확인할 수 있었습니다. 그래서 문제는 stage 2에 존재할 가능성이 높다는 결론을 내릴 수 있었습니다.

![try_1](https://github.com/juni1119/take_home_project/raw/main/try_4.png)
