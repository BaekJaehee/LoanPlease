package com.d105.loanplease.domain.game.dto;


public class GameModel {
    public String gender;
    public String car;
    public String reality;
    public int child_num;
    public float income_total;
    public String income_type;
    public String edu_type;
    public String family_type;
    public String house_type;
    public int DAYS_BIRTH;
    public int DAYS_EMPLOYED;
    public int work_phone;
    public int phone;
    public int email;
    public String occyp_type;
    public float family_size;
    public float begin_month;

    public GameModel(String gender, String car, String reality, Integer child_num, Integer income_total, String income_type,
                     String edu_type, String family_type, String house_type, Integer DAYS_BIRTH, Integer DAYS_EMPLOYED,
                     String occyp_type, Integer family_size, Integer begin_month) {
        this.gender = gender;
        this.car = car;
        this.reality = reality;
        this.child_num = child_num;
        this.income_total = Float.valueOf(income_total);
        this.income_type = income_type;
        this.edu_type = edu_type;
        this.family_type = family_type;
        this.house_type = house_type;
        this.DAYS_BIRTH = DAYS_BIRTH;
        this.DAYS_EMPLOYED = DAYS_EMPLOYED;
        this.work_phone = 0;  // Default value
        this.phone = 0;       // Default value
        this.email = 0;       // Default value
        this.occyp_type = occyp_type;
        this.family_size = Float.valueOf(family_size);
        this.begin_month = Float.valueOf(begin_month);
    }
}
