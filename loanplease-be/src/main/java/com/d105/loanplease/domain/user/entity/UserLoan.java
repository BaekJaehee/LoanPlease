package com.d105.loanplease.domain.user.entity;

import com.d105.loanplease.domain.store.domain.Loan;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "user_loan_tb")
public class UserLoan {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_loan_id")
    private Long userLoanId;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;

    @ManyToOne
    @JoinColumn(name = "loan_id")
    private Loan loan;

    public UserLoan(Loan loan, User user) {
        this.user = user;
        user.getUserLoanList().add(this);
        this.loan = loan;
    }

    // 대출 상품 구매
    public static UserLoan purchaseLoan(Loan loan, User user) {

        user.hasLoan(loan.getLoanId()); // 이미 보유한 대출 상품인지 체크

        UserLoan userLoan = new UserLoan(loan, user);
        user.purchase(loan.getPrice()); // 포인트가 부족한지 체크

        return userLoan;
    }
}
