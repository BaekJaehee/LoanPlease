package com.d105.loanplease.store;

import com.d105.loanplease.domain.store.application.port.in.LoanUseCase;
import com.d105.loanplease.domain.store.application.port.out.LoanPort;
import com.d105.loanplease.domain.store.application.service.LoanService;
import com.d105.loanplease.domain.store.domain.Loan;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.entity.UserLoan;
import com.d105.loanplease.domain.user.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

@SpringBootTest
@ExtendWith(MockitoExtension.class)
public class LoanServiceTest {

    @Autowired
    private LoanService loanService;

    @Autowired
    private UserRepository userRepository;

    @Test
    void 대출상품전체조회() {
        List<Loan> loanList = loanService.inquiryAllLoans();
        assertThat(loanList.size()).isEqualTo(2);
    }

    @Test
    void 특정대출상품조회() {
        final Long firstLoanId = 1L;
        Loan firstLoan = loanService.inquiryLoanById(firstLoanId);

        assertThat(firstLoan.getContent()).isEqualTo("이 상품은 첫번째 테스트용 대출 상품입니다");
        assertThat(firstLoan.getInterest()).isEqualTo(0.1);

        final Long secondLoanId = 2L;
        Loan secondLoan = loanService.inquiryLoanById(secondLoanId);

        assertThat(secondLoan.getContent()).isEqualTo("이 상품은 두번째 테스트용 대출 상품입니다");
        assertThat(secondLoan.getInterest()).isEqualTo(0.2);
    }

    @Test
    void 대출상품구매() {
        final Long loanId = 1L;
        final Long userId = 1L;
        loanService.purchaseLoan(loanId);

        User user = userRepository.findById(userId)
                .orElseThrow(() -> new IllegalArgumentException("없는 유저입니다."));

        assertThat(user.getUserLoanList().size()).isEqualTo(1);
        assertThat(user.getUserLoanList().get(0).getLoan().getName()).isEqualTo("테스트용 대출1");
    }
}
