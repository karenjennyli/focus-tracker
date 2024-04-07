import {
    Box,
    Button,
    Container,
    Flex,
    Heading,
    Image,
    Text,
  } from '@chakra-ui/react';
import {useNavigate} from 'react-router-dom';
import currentSession from './current_session_image.png'
import sessionSummary from './session_summary_image.png'
  
  function FeatureComponent() {
    const navigate = useNavigate();
  
    return (
      <Container maxW="container.xl" py={10}>
        <Flex direction={{ base: "column", lg: "row" }} align="center" justify="space-between" gap={20}>
          
          {/* Left side - Text content */}
          <Box flex="1" color="white">
            <Heading as="h1" size="3xl" fontWeight="bold" mb={4}>
              Real-Time data visualization and analytics
            </Heading>
            <Text fontSize="lg" mb={6}>
              Immediate distraction feedback and interactive graphs provided
            </Text>
            <Button colorScheme="blue" size="lg" px={10} py={6} onClick={() => navigate('/calibration-page')}>
              Get started
            </Button>
          </Box>
  
          {/* Right side - Overlapping images */}
          <Box position="relative" width={{ base: "100%", lg: "30%" }} mr={{ lg: "20%", xl: "20%" }}>
            <Image
              src={currentSession}
              alt="Illustration of connection"
              position="relative"
              zIndex="1"
            />
            <Image
              src={sessionSummary}
              alt="Illustration of transaction"
              position="absolute"
              right="-80%"
              top="50%"
              transform="translateY(-10%)"
              zIndex="0"
              width="100%" // Used to adjust the width
            />
          </Box>
  
        </Flex>
      </Container>
    // <Container maxW="container.xl" py={10}>
    //     <Flex direction={{ base: "column", lg: "row" }} align="center" justify="space-between" gap={10}>
            
    //         {/* Left side - Text content */}
    //         <Box flex="1" color="white">
    //         <Heading as="h1" size="3xl" fontWeight="bold" mb={4}>
    //             Real-Time data visualization and analytics
    //         </Heading>
    //         <Text fontSize="lg" mb={6}>
    //             Onboard every user, connect to any wallet, and build apps that anyone can use — with in-app wallets, account abstraction, and fiat & crypto payments.
    //         </Text>
    //         <Button colorScheme="blue" size="lg" px={10} py={6} onClick={() => navigate('/calibration-page')}>
    //             Get started
    //         </Button>
    //         </Box>

    //         {/* Right side - Overlapping images */}
    //         <Box position="relative" width={{ base: "100%", lg: "40%" }} minHeight="36rem">
    //         <Image
    //             src={currentSession} // Replace with the path to your first image
    //             alt="Illustration of connection"
    //             position="relative"
    //             zIndex="1"
    //             width="full" // Makes sure the image takes the full width of the container
    //         />
    //         <Image
    //             src={sessionSummary} // Replace with the path to your second image
    //             alt="Illustration of transaction"
    //             position="absolute"
    //             right={{ base: "0", lg: "-20%", xl: "-60%" }} // Adjusted for different breakpoints
    //             top="70%"
    //             transform="translateY(-50%)" // Adjusts for the full translation on Y-axis
    //             zIndex="0"
    //             width={{ base: "80%", lg: "100%" }} // Adjust the width based on breakpoints
    //         />
    //         </Box>

    //     </Flex>
    // </Container>
    );
  }
  
  export default FeatureComponent;
  