package com.d105.loanplease.domain.game.serviceImpl;

import com.d105.loanplease.domain.game.Fields.*;
import com.d105.loanplease.domain.game.dto.*;
import com.d105.loanplease.domain.game.exception.AIException;
import com.d105.loanplease.domain.store.adapter.out.LoanRepository;
import com.d105.loanplease.domain.store.domain.Loan;
import com.d105.loanplease.domain.user.dto.UserItemResDto;
import com.d105.loanplease.domain.user.entity.UserItem;
import com.d105.loanplease.domain.user.repository.UserItemRepository;
import com.d105.loanplease.domain.game.response.GameInfoResponse;
import com.d105.loanplease.domain.game.response.ResultResponse;
import com.d105.loanplease.domain.game.response.ScoreResponse;
import com.d105.loanplease.domain.game.service.GameService;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.global.util.SecurityUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Service
public class GameServiceImpl implements GameService {

    @Autowired
    private LoanRepository loanRepository;

    @Autowired
    private UserItemRepository userItemRepository;

    @Autowired
    private SecurityUtil securityUtil;
    @Override
    public ResponseEntity<GameInfoResponse> getGameInfo() {

        // 랜덤으로 값을 생성한다.
        // Random 객체 생성
        Random random = new Random();
        int randomIndex;
        int min;    // 최소값
        int max;  // 최대값


        /**
         * 가족 유형
         * */
        // family_type
        FamilyType[] familyTypes = FamilyType.values();
        randomIndex = random.nextInt(familyTypes.length);
        FamilyType familyType = familyTypes[randomIndex];

        // family size : 1.0 ~ 4.0
        min = 1;
        max = 4;
        if(familyType==FamilyType.CIVIL || familyType==FamilyType.MARRIED)
            min=2;

        int familySize = random.nextInt((max-min)+1)+min;

        // child num : 0~4
        int childNum = familySize-2;
        if(familyType==FamilyType.SEPERATED && childNum>0){
            min = 0; max = childNum;
            childNum = random.nextInt((max-min)+1)+min;
        }
        if(familyType==FamilyType.SINGLE){
            familySize = 1;
            childNum = 0;
        }

        if(childNum<0) childNum=0;

        /**
        * 직업 유형
         * **/
        // occyp_type
        OccypType[] occypTypes = OccypType.values();
        randomIndex = random.nextInt(occypTypes.length);
        OccypType occypType = occypTypes[randomIndex];

        // income total : 27000.0 ~ 1575000
        min = 300000;
        if(occypType==OccypType.HIGHSKILL || occypType==OccypType.MEDICINESTAFF || occypType==OccypType.ACCOUNTANTS){
            min = 1000000;
        }
        if(occypType==OccypType.CORESTAFF || occypType==OccypType.MANAGERS) min = 700000;
        max = 1575000;
        if(occypType==OccypType.LABORERS || occypType==OccypType.LOWSKILL) max = 600000;

        if(occypType==OccypType.STUDENT){
            min = 60000;
            max = 150000;
        }

        int incomeTotal = random.nextInt((max-min)+1)+min;



        // DAYS_BIRTH 7705 ~ 25152
        min = 7705;
        max = 25152;
        int daysBirth = random.nextInt((max-min)+1)+min;

        // DAYS_EMPLOYED 0 ~ 8000
        min = 0;
        max = 8000;
        int daysEmployed = random.nextInt((max-min)+1)+min;

        // begin_month : 0 ~ 60
        min = 0;
        max = 60;
        int beginMonth = random.nextInt((max-min)+1)+min;

        // income_type
        IncomeType[] incomeTypes = IncomeType.values();
        randomIndex = random.nextInt(incomeTypes.length);
        IncomeType incomeType = incomeTypes[randomIndex];

        // edu_type
        EduType[] eduTypes = EduType.values();
        randomIndex = random.nextInt(eduTypes.length);
        EduType eduType = eduTypes[randomIndex];

        if(incomeType==IncomeType.STUDENT){
            eduType = EduType.HIGHER;
            randomIndex = random.nextInt(2);
            occypType = occypTypes[randomIndex];
        }
        if(occypType==OccypType.LOWSKILL || occypType==OccypType.LABORERS){
            incomeType = IncomeType.WORKING;
        }

        // house_type
        HouseType[] houseTypes = HouseType.values();
        randomIndex = random.nextInt(houseTypes.length);
        HouseType houseType = houseTypes[randomIndex];


        // YN 여부
        YN[] yns = YN.values();
        randomIndex = random.nextInt(yns.length);
        YN car = yns[randomIndex];
        randomIndex = random.nextInt(yns.length);
        YN reality = yns[randomIndex];

        FinancialInfo financialInfo = new FinancialInfo(incomeTotal/100, incomeType.getKoreanName(), occypType.getKoreanName());
        NonFinancialInfo nonFinancialInfo = new NonFinancialInfo(car.getKoreanName(), reality.getKoreanName(), childNum, eduType.getKoreanName(), familyType.getKoreanName(), houseType.getKoreanName(), daysEmployed, familySize);


        // 데이터를 보낸다.
        Purpose[] purposes = Purpose.values();
        randomIndex = random.nextInt(purposes.length);
        Purpose purpose = purposes[randomIndex];
        LoanRequest loanRequest = new LoanRequest(3, 4, purpose.getPurposeKorean());  // 랜덤 작업 필요

        int age = daysBirth/360;
        GenderType[] genders = GenderType.values();
        randomIndex = random.nextInt(genders.length);
        GenderType gender = genders[randomIndex];

        String[] lastNames = new String[]{"김", "이", "백", "정", "최", "남", "박", "홍", "우", "한", "금", "오", "조"};
        randomIndex = random.nextInt(lastNames.length);
        String name = lastNames[randomIndex];

        if(gender.getKoreanName().equals("남성")){
            String[] firstNames = new String[]{"민수", "인호", "민우", "중원", "창영", "수현", "유준", "하빈", "호성", "철주", "현직", "남현"};
            randomIndex = random.nextInt(firstNames.length);
            if(firstNames[randomIndex].equals("중원")) name = "이";    // 이중원의 요청 ^_^
            name += firstNames[randomIndex];
        }else{
            String[] firstNames = new String[]{"재희", "설연", "수진", "채연", "규리", "예인", "지수", "수연", "유리", "소현", "세림"};
            randomIndex = random.nextInt(firstNames.length);
            name += firstNames[randomIndex];
        }



        int picNumber = selectPicNumber(age, gender.getKoreanName());

        List<Boolean> materials = new ArrayList<>();
        for(int i=0; i<2; i++){
            int probability = random.nextInt(10);
            // 10% 확률로 준비물 false
            if(probability == 0) materials.add(false);
            else materials.add(true);
        }


        CustomerInfo customerInfo = new CustomerInfo(name, age, gender.getKoreanName(), picNumber, purpose.getPurposeKorean()+"목적으로 대출해주세요!", materials);


        // 영문명을 활용하여
        // Fast API에 결과를 전송한다.
        GameModel gmaeModel = new GameModel(gender.getEnglishName(), car.getEnglishName(), reality.getEnglishName(),
                childNum, incomeTotal, incomeType.getEnglishName(), eduType.getEnglishName(), familyType.getEnglishName(),
                houseType.getEnglishName(), daysBirth, daysEmployed, occypType.getEnglishName(), familySize, beginMonth);

        int credit;

        try{
            credit = getCreditFromAI(gmaeModel);
        }catch(Exception e){
            throw new AIException("AI 오류입니다");
        }

        if(incomeTotal/100 > 8000 || reality.getEnglishName().equals("Y")) credit--;

        if(incomeType.getKoreanName().equals("공기업") || occypType.getKoreanName().equals("의료계") || occypType.getKoreanName().equals("CEO")
        || occypType.getKoreanName().equals("고급 기술자") || occypType.getKoreanName().equals("회계사")
        || occypType.getKoreanName().equals("부동산 중개인") || occypType.getKoreanName().equals("공무원")) credit--;

        if(credit<0) credit = 0;
        GameInfo gameInfo = new GameInfo(loanRequest, customerInfo, financialInfo, nonFinancialInfo, credit);
        GameInfoResponse response = GameInfoResponse.createGameInfoResponse(HttpStatus.OK.value(), "게임 정보를 성공적으로 받아왔습니다.", gameInfo);

        return ResponseEntity.ok(response);
    }

    @Override
    public ResponseEntity<ScoreResponse> getAwayCustomer(GameInfo gameInfo) {
        Score score;
        // 준비물이 모두 갖춰졌다면 점수 감점
        if(!gameInfo.getCustomerInfo().getCustomerMaterials().contains(false)){
            score = new Score(-200, "왜 안해주세요ㅡㅡ", "준비물이 모두 있습니다.");
        }else{
            score = new Score(200, "아 맞다", "준비물이 부족합니다.");
        }

        ScoreResponse response = ScoreResponse.createScoreResponse(HttpStatus.OK.value(), "점수를 획득했습니다.", score);
        return ResponseEntity.ok(response);
    }

    @Override
    public ResponseEntity<ScoreResponse> gainScore(int num, GameInfo gameInfo) {
        Score score;
        if(gameInfo.getCustomerInfo().getCustomerMaterials().contains(false)){
            score = new Score(-200, "ㅋ 이걸 해주네", "준비물이 부족합니다.");
        }else{
            score = calculateScore(num, gameInfo);
        }

        ScoreResponse response = ScoreResponse.createScoreResponse(HttpStatus.OK.value(), "점수를 획득했습니다.", score);
        return ResponseEntity.ok(response);
    }

    @Override
    @Transactional
    public ResponseEntity<ResultResponse> saveScore(int score) {
        // User의 정보를 불러옵니다.
        User user = securityUtil.getCurrentUserDetails();

        // 점수가 최고 점수라면 갱신합니다.
        if(user.getScore() < score){
            user.setScore(score);
        }
        int point = (int)(score*(0.1));
        if(score<5000) point=500;   // 기본 포인트
        user.setPoint(user.getPoint()+point);


        ResultResponse response = ResultResponse.createResultResponse(HttpStatus.OK.value(), "점수를 저장했습니다.", point);

        return ResponseEntity.ok(response);
    }

    private int selectPicNumber(int age, String gender){
        int generation = age/10;
        int min; int max;
        Random random = new Random();
        int[] womanPictures = new int[]{1,2,3,4,5,6,7,8, 9};
        int[] manPictures = new int[]{11,12,13,14,15,16,17,18};

        switch (gender){
            case "남성":
                if(generation <3) return manPictures[0];
                else if(generation <4){
                    min = 1; max = 2;
                    return manPictures[random.nextInt(max-min+1)+min];
                }else if (generation<5)
                    return manPictures[3];
                else if(generation<6) return manPictures[4];
                else{
                    min=5; max=7;
                    return manPictures[random.nextInt(max-min+1)+min];
                }
            case "여성":
                if(generation <3){
                    min = 0; max = 3;
                    return womanPictures[random.nextInt(max-min+1)];
                }else if(generation <4){
                    min = 4; max = 5;
                    return womanPictures[random.nextInt(max-min+1)+min];
                }else if (generation<5){
                    return womanPictures[6];
                }else if(generation<6) return womanPictures[7];
                else return womanPictures[8];
        }
        return 1;
    }

    private Score calculateScore(int num, GameInfo gameInfo){
        Loan loan = loanRepository.getReferenceById((long)num);

//         최소 신용 최대 신용과 차이가 적은걸 반영하기.
        int diff = (Math.abs(gameInfo.getCredit()-loan.getMinCredit()));
        diff = Math.min(diff, Math.abs(gameInfo.getCredit()-loan.getMaxCredit()));

        if(loan.getMinCredit()==0 && loan.getMaxCredit()==2) diff = 0;

//        if(!isContainsCredit(gameInfo.getCredit(), loan.getMinCredit(), loan.getMaxCredit())){
//            return new Score(-500, "후엥", "대출상품에 필요한 신용 등급과 일치하지 않습니다.");
//        }

        int defaultScore = 500 + (-100)*diff;
//        int defaultScore = 500;


        if(num==1 || num==2 || num==3){
            // 기본 저금리
            int score = (int)(defaultScore*(1+loan.getInterest()));
            return new Score(score, "감사합니다!", "적절한 신용도의 대출을 추천했습니다.");

        }else if(num==4){
            // 히포크라테스 대출
            int score = (int)(defaultScore*(1+loan.getInterest()));
            String reason;
            if(gameInfo.getFinancialInfo().getOccypType().equals(OccypType.MEDICINESTAFF.getKoreanName())){
                score += 200;
                reason = "의료계 종사자 조건과 일치합니다.";
            }else reason = "의료계 종사자가 아닙니다.";

            return new Score(score, "감사합니다!!", reason);
        }else if(num==5){
            // 녹봉이요 대출
            int score = (int)(defaultScore*(1+loan.getInterest()));
            String reason;
            if(gameInfo.getFinancialInfo().getOccypType().equals(OccypType.GONG.getKoreanName())
            || gameInfo.getFinancialInfo().getIncomeType().equals(IncomeType.STATE.getKoreanName())){
                score += 200;
                reason = "공무 수행 종사자 조건과 일치합니다.";
            }else reason = "공무 수행 종사자가 아닙니다.";

            return new Score(score, "감사합니다!!", reason);
        }else if(num==6){
            // 저세상갓숭 대출
            int score = (int)((defaultScore-200)*(1+loan.getInterest()));

            return new Score(score, "너무 금리가 높은거 아닌가요?! ㅡㅡ", "금리가 너무 높아 화가 나서 감점이 적용되었습니다.");

        }else if(num==7){
            // 고오-급 대출
            String reason;
            int score = defaultScore*(int)(1+loan.getInterest());
            double incomeTotal = gameInfo.getFinancialInfo().getIncomeTotal();
            if(incomeTotal >= 100000000){
                score += (int)(incomeTotal*(0.01));
                reason = "연봉 1억 이상인 고소득자 조건과 일치합니다.";
            }else{
                score += (int)(incomeTotal*(0.005));
                reason = "연봉 1억 이상인 고소득자 조건과 일치하지 않습니다.";
            }

            return new Score(score, "감사합니다!", reason);

        }else if(num==8){
            // 소중한 상생 대출
            String reason = "소상공인 조건을 만족하지 않았습니다.";
            int score = defaultScore*(int)(1+loan.getInterest());
            if(gameInfo.getFinancialInfo().getIncomeTotal() < 100000000
            && !gameInfo.getFinancialInfo().getIncomeType().equals(IncomeType.STUDENT.getKoreanName())
            && !gameInfo.getFinancialInfo().getIncomeType().equals(IncomeType.STATE.getKoreanName())){
                score += 150;
                reason = "소상공인 조건을 만족했습니다.";
            }

            return new Score(score, "감사합니다!", reason);
        }else if(num==9){
            // 거인의 대출
            String reason = "저신용자 청년 조건을 충족하지 않았습니다.";
            int score = defaultScore*(int)(1+loan.getInterest());
            if(gameInfo.getCustomerInfo().getAge()/10 == 2
            && gameInfo.getCredit() > 0){
                score += 150;
                reason = "저신용자 청년 조건을 충족했습니다.";

                if(gameInfo.getCustomerInfo().getName().equals("이중원")){
                    score -=150;
                    score+=30000;
                    reason = "히든 케이스 충족!!! 보너스 3만점입니다!!!";
                }
            }
            return new Score(score, "감사해요~", reason);


        }else if(num==10){
            // 취업하고 싶은 코린이 대출
            String reason = "취업하고 싶은 코린이 조건을 충족하지 않았습니다.";
            int score = defaultScore*(int)(1+loan.getInterest());
            if(gameInfo.getFinancialInfo().getIncomeType().equals(IncomeType.STUDENT.getKoreanName())
            && gameInfo.getFinancialInfo().getOccypType().equals(OccypType.IT.getKoreanName())
            && gameInfo.getCustomerInfo().getAge()/10 == 2){
                score += 2500;
                reason = "20대 IT직무를 준비하는 학생 조건을 충족했습니다.";
            }
            return new Score(score, "감사해요~", reason);
        }else throw new IllegalArgumentException("상품 정보가 없거나 게임정보가 없습니다.");
    }
    private boolean isContainsCredit(int credit, int minCredit, int maxCredit){
        if(minCredit <= credit && credit <= maxCredit) return true;
        return false;
    }

    private int getCreditFromAI(GameModel gameModel) throws JsonProcessingException {
        RestTemplate restTemplate = new RestTemplate();
        String url = "http://loanplease.kr:8000/model";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(gameModel);

        HttpEntity<String> request = new HttpEntity<>(json, headers);

        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);

        JsonNode root = mapper.readTree(response.getBody());
        int credit = root.path("credit").asInt();

        return credit;
    }

    @Override
    @Transactional
    public ResponseEntity<UseItemResponse> useItem(Long userItemId) {
        User user = securityUtil.getCurrentUserDetails();
        UserItem userItem = userItemRepository.findById(userItemId)
                .orElseThrow(() -> new IllegalArgumentException("해당 아이템이 없습니다."));

        userItem.useItem(); // 아이템 사용

        List<UserItemResDto> userItemResDtoList = new ArrayList<>();
        List<UserItem> userItemList = user.getUserItemList();

        for(UserItem uItem: userItemList) {
            userItemResDtoList.add(new UserItemResDto(uItem));
        }

        UseItemResponse response = UseItemResponse.builder()
                .userItemResDtoList(userItemResDtoList)
                .build();

        return ResponseEntity.ok(response);
    }
}
