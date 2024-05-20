package com.d105.loanplease.domain.store.domain;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "loan_tb")
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Loan {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "loan_id")
    private Long loanId;

    private String name;
    private String content;
    private Double interest;
    private Integer period;
    private Integer price;
    private String color;
    private Long limitAmount; // limit이 예약어라 계속 오류남
    @Column(name = "min_credit")
    private Integer minCredit;
    @Column(name = "max_credit")
    private Integer maxCredit;

}
